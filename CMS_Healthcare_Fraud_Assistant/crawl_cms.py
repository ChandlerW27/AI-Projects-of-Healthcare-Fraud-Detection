from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from urllib.parse import quote_plus

import requests

from src.helper import DEFAULT_HEADERS, download_cms_source, extract_search_results, polite_delay, save_json

DEFAULT_SEED_FILE = Path("cms_urls.txt")
DEFAULT_OUTPUT = Path("data/cms_documents.json")

SEARCH_QUERIES = [
    "healthcare fraud", "Medicare fraud abuse", "Medicaid program integrity",
    "DMEPOS fraud", "hospice fraud", "provider compliance", "improper payments",
    "medical review audit", "identity theft Medicare", "billing fraud coding",
    "reporting fraud", "predictive analytics fraud", "Marketplace fraud",
]


def read_seed_urls(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            items.append({"title": "CMS curated source", "url": line, "depth": 0})
    return items


def collect_search_candidates(session: requests.Session, pages_per_query: int, results_per_page: int) -> list[dict]:
    candidates: list[dict] = []
    for query in SEARCH_QUERIES:
        for page in range(pages_per_query):
            search_url = f"https://www.cms.gov/search/cms?keys={quote_plus(query)}&page={page}"
            print(f"Search: {query!r}, page {page + 1}")
            try:
                for item in extract_search_results(session, search_url, results_per_page):
                    item["depth"] = 0
                    candidates.append(item)
            except Exception as exc:
                print(f"  Search page skipped: {exc}")
            polite_delay(0.15)
    return candidates


def crawl(pages_per_query: int, results_per_page: int, max_documents: int, link_depth: int, output: Path) -> None:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    queue = deque(read_seed_urls(DEFAULT_SEED_FILE))
    queue.extend(collect_search_candidates(session, pages_per_query, results_per_page))
    queued = {item["url"] for item in queue}
    visited: set[str] = set()
    documents: list[dict] = []

    while queue and len(documents) < max_documents:
        item = queue.popleft()
        url = item["url"]
        depth = int(item.get("depth", 0))
        if url in visited:
            continue
        visited.add(url)
        print(f"[{len(documents) + 1}/{max_documents}] depth={depth} {url}")

        try:
            document = download_cms_source(session, url, item.get("title", "CMS source"))
            if not document:
                print("  Skipped: too little readable text")
                continue
            linked_sources = document.pop("linked_sources", [])
            documents.append(document)
            print(f"  Added {document['source_type']}: {len(document['text']):,} characters")

            if depth < link_depth:
                for linked in linked_sources:
                    linked_url = linked["url"]
                    if linked_url not in visited and linked_url not in queued:
                        linked["depth"] = depth + 1
                        queue.append(linked)
                        queued.add(linked_url)
        except Exception as exc:
            print(f"  Skipped: {exc}")
        polite_delay(0.2)

    save_json(documents, output)
    print(f"\nSaved {len(documents)} CMS documents to {output.resolve()}")
    print(f"Visited {len(visited)} candidate URLs; {len(queue)} candidates remain unprocessed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Broad CMS.gov fraud webpage and PDF crawler.")
    parser.add_argument("--pages-per-query", type=int, default=2)
    parser.add_argument("--results-per-page", type=int, default=12)
    parser.add_argument("--max-documents", type=int, default=140)
    parser.add_argument("--link-depth", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    crawl(args.pages_per_query, args.results_per_page, args.max_documents, args.link_depth, args.output)
