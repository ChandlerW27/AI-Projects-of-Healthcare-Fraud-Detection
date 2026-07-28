from __future__ import annotations

import json
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from config import (DENSE_CANDIDATES, EMBEDDING_MODEL, FINAL_CONTEXT_K, FLASK_PORT,
                    LEXICAL_CANDIDATES, MAX_CONTEXT_CHARS, OPENAI_API_KEY,
                    OPENAI_MODEL, PINECONE_API_KEY, PINECONE_INDEX, PINECONE_NAMESPACE)
from src.helper import get_embeddings, tokenize
from src.prompt import SYSTEM_PROMPT

app = Flask(__name__)
CHUNKS_FILE = Path("data/allergy_chunks.json")

EMERGENCY_PATTERNS = [
    r"trouble breathing", r"can't breathe", r"cannot breathe", r"throat (?:is )?(?:closing|swelling)",
    r"tongue swelling", r"blue lips", r"faint(?:ed|ing)?", r"passed out", r"severe wheez",
    r"anaphyl", r"epipen", r"epinephrine", r"rapidly worsening",
]


def require_settings() -> None:
    missing = [k for k, v in {"PINECONE_API_KEY": PINECONE_API_KEY, "OPENAI_API_KEY": OPENAI_API_KEY}.items() if not v]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("Missing data/allergy_chunks.json. Run crawl_allergy.py and store_index.py first.")


require_settings()
embeddings = get_embeddings(EMBEDDING_MODEL)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=PINECONE_NAMESPACE)
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
records = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
bm25 = BM25Okapi([tokenize(f"{x['metadata'].get('title','')} {x['text']}") for x in records])


def emergency_message(question: str) -> str | None:
    lowered = question.lower()
    if any(re.search(pattern, lowered) for pattern in EMERGENCY_PATTERNS):
        return (
            "These symptoms may represent a medical emergency. If there is trouble breathing, "
            "throat or tongue swelling, fainting, blue lips, severe wheezing, or a rapidly worsening "
            "reaction: use prescribed epinephrine immediately and call 911 (or your local emergency "
            "number). Do not wait for the chatbot or rely on antihistamines alone."
        )
    return None


def expand_question(question: str) -> str:
    q = question.strip()
    lower = q.lower()
    additions = []
    if any(x in lower for x in ["flu", "cold", "virus", "respiratory infection"]):
        additions.append("allergic rhinitis versus common cold influenza COVID symptom timing fever itching body aches testing")
    if any(x in lower for x in ["test", "diagnosis", "lab"]):
        additions.append("skin prick test specific IgE blood test patch test oral food challenge interpretation false positives")
    if any(x in lower for x in ["shot", "immunotherapy", "slit"]):
        additions.append("allergen immunotherapy allergy shots sublingual tablets indications risks duration allergist")
    if any(x in lower for x in ["food", "eat", "peanut", "milk", "egg"]):
        additions.append("food allergy symptoms anaphylaxis epinephrine label avoidance diagnosis oral challenge")
    return f"{q}. {' '.join(additions)}" if additions else q


def retrieve(query: str) -> list[Document]:
    dense = vector_store.similarity_search(query, k=DENSE_CANDIDATES)
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:LEXICAL_CANDIDATES]
    lexical = [Document(page_content=records[i]["text"], metadata=records[i]["metadata"]) for i in top if scores[i] > 0]
    fused = {}
    for docs, weight in ((dense, 1.0), (lexical, 1.15)):
        for rank, doc in enumerate(docs, 1):
            key = doc.metadata.get("vector_id") or f"{doc.metadata.get('url')}|{doc.metadata.get('chunk_id')}"
            entry = fused.setdefault(key, {"doc": doc, "score": 0, "methods": 0})
            entry["score"] += weight / (60 + rank)
            entry["methods"] += 1
    ranked = sorted(fused.values(), key=lambda x: (x["methods"], x["score"]), reverse=True)
    return [x["doc"] for x in ranked[:FINAL_CONTEXT_K]]


def context_and_sources(documents: list[Document]) -> tuple[str, list[dict]]:
    blocks, sources, seen, total = [], [], set(), 0
    for number, doc in enumerate(documents, 1):
        title, url = doc.metadata.get("title", "Allergy source"), doc.metadata.get("url", "")
        source = doc.metadata.get("source", "Trusted health source")
        block = f"[Source {number}]\nOrganization: {source}\nTitle: {title}\nURL: {url}\nText: {doc.page_content}"
        if total + len(block) > MAX_CONTEXT_CHARS and blocks:
            break
        blocks.append(block); total += len(block)
        if url and url not in seen:
            sources.append({"number": number, "title": title, "url": url, "source": source})
            seen.add(url)
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
    urgent = emergency_message(question)
    if urgent:
        return jsonify({"answer": urgent, "sources": [], "urgent": True})
    try:
        query = expand_question(question)
        documents = retrieve(query)
        context, sources = context_and_sources(documents)
        response = llm.invoke([
            ("system", SYSTEM_PROMPT.format(context=context)),
            ("human", f"User question: {question}\nExpanded retrieval query: {query}"),
        ])
        return jsonify({"answer": response.content, "sources": sources, "urgent": False})
    except Exception as exc:
        app.logger.exception("Chat request failed")
        return jsonify({"error": f"The chatbot could not answer: {exc}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "index": PINECONE_INDEX, "namespace": PINECONE_NAMESPACE,
                    "local_chunks": len(records), "retrieval": "hybrid dense + BM25 + RRF"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True)
