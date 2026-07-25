from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from config import (
    DENSE_CANDIDATES, EMBEDDING_MODEL, FINAL_CONTEXT_K, FLASK_PORT,
    LEXICAL_CANDIDATES, MAX_CONTEXT_CHARS, OPENAI_API_KEY, OPENAI_MODEL,
    PINECONE_API_KEY, PINECONE_INDEX, PINECONE_NAMESPACE,
)
from src.helper import get_embeddings, tokenize
from src.prompt import SYSTEM_PROMPT

app = Flask(__name__)
CHUNKS_FILE = Path("data/cms_chunks.json")


def require_settings() -> None:
    missing = [name for name, value in {
        "PINECONE_API_KEY": PINECONE_API_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY,
    }.items() if not value]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("Missing data/cms_chunks.json. Run crawl_cms.py and store_index.py first.")


require_settings()
embeddings = get_embeddings(EMBEDDING_MODEL)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=PINECONE_NAMESPACE)
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)

chunk_records = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
chunk_by_id = {item["id"]: item for item in chunk_records}
bm25 = BM25Okapi([tokenize(f"{x['metadata'].get('title', '')} {x['text']}") for x in chunk_records])


def expand_question(question: str) -> str:
    q = question.strip()
    lowered = q.lower().rstrip(" ?.!;")
    broad = {
        "how to detect fraud", "how to detect the fraud", "fraud detection",
        "how can fraud be detected", "how do i detect fraud", "detect healthcare fraud",
    }
    if lowered in broad or len(tokenize(q)) <= 4 and "fraud" in lowered:
        return (
            f"{q}. CMS healthcare fraud waste and abuse warning signs, suspicious billing patterns, "
            "data analytics, audits, medical review, provider enrollment controls, prevention, and reporting."
        )
    synonyms = {
        "dme": "DMEPOS durable medical equipment prosthetics orthotics supplies",
        "report fraud": "report suspected Medicare Medicaid healthcare fraud waste abuse",
        "fake bill": "billing for services or items not provided false claims",
    }
    additions = [value for key, value in synonyms.items() if key in lowered]
    return f"{q}. {' '.join(additions)}" if additions else q


def dense_candidates(query: str) -> list[Document]:
    return vector_store.similarity_search(query, k=DENSE_CANDIDATES)


def lexical_candidates(query: str) -> list[Document]:
    scores = bm25.get_scores(tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:LEXICAL_CANDIDATES]
    return [Document(page_content=chunk_records[i]["text"], metadata=chunk_records[i]["metadata"]) for i in top_indices if scores[i] > 0]


def reciprocal_rank_fusion(dense: list[Document], lexical: list[Document]) -> list[Document]:
    fused: dict[str, dict] = {}
    for source_name, documents, weight in (("dense", dense, 1.0), ("lexical", lexical, 1.15)):
        for rank, doc in enumerate(documents, start=1):
            vector_id = doc.metadata.get("vector_id")
            if not vector_id:
                url = doc.metadata.get("url", "")
                chunk_id = doc.metadata.get("chunk_id", 0)
                vector_id = f"{url}|{chunk_id}"
            entry = fused.setdefault(vector_id, {"document": doc, "score": 0.0, "methods": set()})
            entry["score"] += weight / (60 + rank)
            entry["methods"].add(source_name)
    ranked = sorted(fused.values(), key=lambda x: (len(x["methods"]), x["score"]), reverse=True)
    return [item["document"] for item in ranked[:FINAL_CONTEXT_K]]


def retrieve(question: str) -> tuple[str, list[Document]]:
    search_query = expand_question(question)
    dense = dense_candidates(search_query)
    lexical = lexical_candidates(search_query)
    return search_query, reciprocal_rank_fusion(dense, lexical)


def build_context(documents: list[Document]) -> tuple[str, list[dict]]:
    blocks: list[str] = []
    sources: list[dict] = []
    seen_urls: set[str] = set()
    total_chars = 0
    for number, document in enumerate(documents, start=1):
        title = document.metadata.get("title", "CMS source")
        url = document.metadata.get("url", "")
        block = f"[Source {number}]\nTitle: {title}\nURL: {url}\nText: {document.page_content}"
        if total_chars + len(block) > MAX_CONTEXT_CHARS and blocks:
            break
        blocks.append(block)
        total_chars += len(block)
        if url and url not in seen_urls:
            sources.append({"number": number, "title": title, "url": url})
            seen_urls.add(url)
    return "\n\n".join(blocks), sources


@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    question = (request.form.get("msg") or payload.get("msg") or payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        search_query, documents = retrieve(question)
        context, sources = build_context(documents)
        prompt = SYSTEM_PROMPT.format(context=context)
        response = llm.invoke([
            ("system", prompt),
            ("human", f"Original question: {question}\nRetrieval query: {search_query}"),
        ])
        return jsonify({"answer": response.content, "sources": sources})
    except Exception as exc:
        app.logger.exception("Chat request failed")
        return jsonify({"error": f"The chatbot could not answer: {exc}"}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok", "index": PINECONE_INDEX, "namespace": PINECONE_NAMESPACE,
        "local_chunks": len(chunk_records), "retrieval": "hybrid dense + BM25 + RRF",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True)
