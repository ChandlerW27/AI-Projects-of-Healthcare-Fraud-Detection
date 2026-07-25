from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CMSFraudRAGStarter/2.0; educational project)"
}

RELEVANCE_TERMS = {
    "fraud", "abuse", "waste", "program integrity", "improper payment",
    "compliance", "audit", "medical review", "billing", "claim", "coding",
    "upcoding", "kickback", "identity theft", "dmepos", "hospice",
    "marketplace", "medicaid integrity", "medicare integrity", "reporting fraud",
    "provider enrollment", "predictive analytics", "payment suspension",
}

SKIP_URL_PARTS = (
    "/newsroom/", "/about-cms/", "/contact-cms/", "/search/",
    "javascript:", "mailto:", "tel:", "/files/zip/",
)


def clean_text(text: str) -> str:
    text = re.sub(r"[\u00a0\t\r]+", " ", text or "")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def normalize_cms_url(base_url: str, href: str | None) -> str | None:
    if not href:
        return None
    url = urljoin(base_url, href).split("#", 1)[0]
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc.lower() not in {"cms.gov", "www.cms.gov"}:
        return None
    if any(part in url.lower() for part in SKIP_URL_PARTS):
        return None
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunparse(("https", "www.cms.gov", parsed.path.rstrip("/") or "/", "", query, ""))


def is_relevant(title: str, url: str, anchor_context: str = "") -> bool:
    haystack = f"{title} {url} {anchor_context}".lower()
    return any(term in haystack for term in RELEVANCE_TERMS)


def extract_page_links(html: str, base_url: str, relevant_only: bool = True) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.find("article") or soup.body or soup
    results: list[dict] = []
    seen: set[str] = set()
    for link in main.select("a[href]"):
        title = clean_text(link.get_text(" ", strip=True))
        url = normalize_cms_url(base_url, link.get("href"))
        parent_text = clean_text(link.parent.get_text(" ", strip=True))[:300] if link.parent else ""
        if not url or url in seen:
            continue
        if relevant_only and not is_relevant(title, url, parent_text):
            continue
        results.append({"title": title or "CMS linked source", "url": url})
        seen.add(url)
    return results


def extract_search_results(session: requests.Session, search_url: str, max_results: int = 20, timeout: int = 45) -> list[dict]:
    response = session.get(search_url, timeout=timeout)
    response.raise_for_status()
    return extract_page_links(response.text, search_url, relevant_only=False)[:max_results]


def extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(f"Page {page_number}\n{text}")
    return "\n\n".join(pages)


def extract_html_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else "CMS webpage"
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "button", "svg", "noscript", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    headings_and_text = []
    for element in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        value = clean_text(element.get_text(" ", strip=True))
        if value and len(value) > 2:
            headings_and_text.append(value)
    text = "\n".join(headings_and_text) or clean_text(main.get_text(" ", strip=True))
    return title, clean_text(text)


def download_cms_source(session: requests.Session, url: str, fallback_title: str = "CMS source", timeout: int = 45) -> dict | None:
    response = session.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    final_url = normalize_cms_url(url, response.url) or url
    content_type = response.headers.get("content-type", "").lower()
    is_pdf = final_url.lower().endswith(".pdf") or "application/pdf" in content_type

    if is_pdf:
        text = extract_pdf_text(response.content)
        title = fallback_title
        source_type = "pdf"
        linked_sources: list[dict] = []
    else:
        title, text = extract_html_text(response.text)
        source_type = "webpage"
        linked_sources = extract_page_links(response.text, final_url, relevant_only=True)

    if len(text) < 250:
        return None
    return {
        "title": title or fallback_title,
        "url": final_url,
        "source_type": source_type,
        "text": text,
        "linked_sources": linked_sources,
    }


def save_json(items: Iterable[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(list(items), ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_documents(path: str | Path) -> list[Document]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Document(
            page_content=item["text"],
            metadata={
                "title": item.get("title", "CMS source"),
                "url": item.get("url", ""),
                "source_type": item.get("source_type", "unknown"),
                "source": "CMS.gov",
            },
        )
        for item in records
    ]


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    per_url_counts: dict[str, int] = {}
    for chunk in chunks:
        url = chunk.metadata.get("url", "")
        chunk_id = per_url_counts.get(url, 0)
        chunk.metadata["chunk_id"] = chunk_id
        per_url_counts[url] = chunk_id + 1
    return chunks


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)?", (text or "").lower())


def polite_delay(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)
