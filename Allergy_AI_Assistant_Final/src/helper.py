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

"""try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass"""

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AllergyHealthRAG/1.0; educational patient project)"
}

ALLOWED_DOMAINS = {
    "aaaai.org", "www.aaaai.org", "allergist.aaaai.org",
    "acaai.org", "www.acaai.org",
    "medlineplus.gov", "www.medlineplus.gov",
    "niaid.nih.gov", "www.niaid.nih.gov",
    "cdc.gov", "www.cdc.gov",
    "fda.gov", "www.fda.gov",
    "aafa.org", "www.aafa.org",
    "foodallergy.org", "www.foodallergy.org",
    "mayoclinic.org", "www.mayoclinic.org",
    "my.clevelandclinic.org", "clevelandclinic.org", "www.clevelandclinic.org",
}

RELEVANCE_TERMS = {
    "allerg", "anaphyl", "asthma", "eczema", "urticaria", "hives", "angioedema",
    "rhinitis", "sinus", "pollen", "mold", "dust mite", "pet dander", "food",
    "drug", "medicine", "insect", "sting", "latex", "contact dermatitis",
    "symptom", "diagnos", "test", "ige", "skin prick", "patch test", "challenge",
    "treatment", "antihistamine", "corticosteroid", "nasal spray", "epinephrine",
    "immunotherapy", "allergy shot", "sublingual", "biologic", "avoidance",
    "patient", "experience", "action plan", "find an allergist", "cold", "flu",
    "respiratory virus", "covid", "wheezing", "cough", "runny nose", "sneezing",
}

SKIP_PARTS = (
    "javascript:", "mailto:", "tel:", "/privacy", "/terms", "/donate", "/careers",
    "/press", "/newsroom", "/advertising", "/login", "/signup", "/search?",
    ".zip", ".mp4", ".jpg", ".jpeg", ".png", ".gif", ".svg",
)


def clean_text(text: str) -> str:
    text = re.sub(r"[\u00a0\t\r]+", " ", text or "")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def canonical_url(base_url: str, href: str | None) -> str | None:
    if not href:
        return None
    url = urljoin(base_url, href).split("#", 1)[0]
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":", 1)[0]
    if parsed.scheme not in {"http", "https"} or host not in ALLOWED_DOMAINS:
        return None
    if any(part in url.lower() for part in SKIP_PARTS):
        return None
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunparse(("https", host, parsed.path.rstrip("/") or "/", "", query, ""))


def source_name(url: str) -> str:
    host = urlparse(url).netloc.lower().replace("www.", "")
    mapping = {
        "aaaai.org": "AAAAI", "allergist.aaaai.org": "AAAAI",
        "acaai.org": "ACAAI", "medlineplus.gov": "MedlinePlus",
        "niaid.nih.gov": "NIAID", "cdc.gov": "CDC", "fda.gov": "FDA",
        "aafa.org": "AAFA", "foodallergy.org": "FARE",
        "mayoclinic.org": "Mayo Clinic", "my.clevelandclinic.org": "Cleveland Clinic",
        "clevelandclinic.org": "Cleveland Clinic",
    }
    return mapping.get(host, host)


def relevant(title: str, url: str, context: str = "") -> bool:
    text = f"{title} {url} {context}".lower()
    return any(term in text for term in RELEVANCE_TERMS)


def extract_links(html: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.find("article") or soup.body or soup
    found, seen = [], set()
    for link in main.select("a[href]"):
        title = clean_text(link.get_text(" ", strip=True))
        url = canonical_url(base_url, link.get("href"))
        parent = clean_text(link.parent.get_text(" ", strip=True))[:350] if link.parent else ""
        if not url or url in seen or not relevant(title, url, parent):
            continue
        found.append({"title": title or "Allergy resource", "url": url})
        seen.add(url)
    return found


def extract_html(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else "Allergy webpage"
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "button", "svg", "noscript", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    parts = []
    for element in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        value = clean_text(element.get_text(" ", strip=True))
        if len(value) > 2:
            parts.append(value)
    return title, clean_text("\n".join(parts) or main.get_text(" ", strip=True))


def extract_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages = []
    for number, page in enumerate(reader.pages, 1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(f"Page {number}\n{text}")
    return "\n\n".join(pages)


def download_source(session: requests.Session, url: str, fallback_title: str, timeout: int = 45) -> dict | None:
    response = session.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    final_url = canonical_url(url, response.url) or url
    content_type = response.headers.get("content-type", "").lower()
    is_pdf = final_url.lower().endswith(".pdf") or "application/pdf" in content_type
    if is_pdf:
        title, text, linked = fallback_title, extract_pdf(response.content), []
        kind = "pdf"
    else:
        title, text = extract_html(response.text)
        linked = extract_links(response.text, final_url)
        kind = "webpage"
    if len(text) < 250:
        return None
    return {
        "title": title or fallback_title,
        "url": final_url,
        "source": source_name(final_url),
        "source_type": kind,
        "text": text,
        "linked_sources": linked,
    }


def save_json(items: Iterable[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(list(items), ensure_ascii=False, indent=2), encoding="utf-8")


def load_documents(path: str | Path) -> list[Document]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Document(page_content=x["text"], metadata={
        "title": x.get("title", "Allergy source"), "url": x.get("url", ""),
        "source": x.get("source", source_name(x.get("url", ""))),
        "source_type": x.get("source_type", "unknown"),
    }) for x in records]


def split_documents(documents: list[Document], size: int, overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    counts: dict[str, int] = {}
    for chunk in chunks:
        url = chunk.metadata.get("url", "")
        chunk.metadata["chunk_id"] = counts.get(url, 0)
        counts[url] = counts.get(url, 0) + 1
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
