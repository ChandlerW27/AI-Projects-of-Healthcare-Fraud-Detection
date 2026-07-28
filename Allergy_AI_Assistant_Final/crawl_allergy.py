from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.helper import DEFAULT_HEADERS, download_source, polite_delay, save_json

SEED_FILE = Path("allergy_urls.txt")
OUTPUT = Path("data/allergy_documents.json")


def read_seeds(path: Path) -> list[dict]:
    seeds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            seeds.append({"title": "Curated allergy resource", "url": line, "depth": 0})
    return seeds


def crawl(max_documents: int, link_depth: int, per_domain_limit: int, output: Path) -> None:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    queue = deque(read_seeds(SEED_FILE))
    queued = {x["url"] for x in queue}
    visited, documents, domain_counts = set(), [], {}

    while queue and len(documents) < max_documents:
        item = queue.popleft()
        url, depth = item["url"], int(item.get("depth", 0))
        domain = urlparse(url).netloc.lower()
        if url in visited or domain_counts.get(domain, 0) >= per_domain_limit:
            continue
        visited.add(url)
        print(f"[{len(documents)+1}/{max_documents}] depth={depth} {url}")
        try:
            document = download_source(session, url, item.get("title", "Allergy resource"))
            if not document:
                print("  skipped: too little readable text")
                continue
            links = document.pop("linked_sources", [])
            documents.append(document)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            print(f"  added {document['source']} {document['source_type']}: {len(document['text']):,} chars")
            if depth < link_depth:
                for linked in links:
                    if linked["url"] not in visited and linked["url"] not in queued:
                        linked["depth"] = depth + 1
                        queue.append(linked)
                        queued.add(linked["url"])
        except Exception as exc:
            print(f"  skipped: {exc}")
        polite_delay(0.25)

    save_json(documents, output)
    print(f"\nSaved {len(documents)} documents to {output.resolve()}")
    print("By domain:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl trusted human-allergy webpages and linked PDFs.")
    parser.add_argument("--max-documents", type=int, default=250)
    parser.add_argument("--link-depth", type=int, default=2)
    parser.add_argument("--per-domain-limit", type=int, default=45)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    crawl(args.max_documents, args.link_depth, args.per_domain_limit, args.output)
