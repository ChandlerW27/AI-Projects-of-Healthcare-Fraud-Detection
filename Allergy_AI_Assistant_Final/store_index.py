from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import (CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIMENSION, EMBEDDING_MODEL,
                    PINECONE_API_KEY, PINECONE_CLOUD, PINECONE_INDEX,
                    PINECONE_NAMESPACE, PINECONE_REGION)
from src.helper import get_embeddings, load_documents, save_json, split_documents

DATA_FILE = Path("data/allergy_documents.json")
CHUNKS_FILE = Path("data/allergy_chunks.json")


def vector_id(url: str, chunk_id: int) -> str:
    return hashlib.sha256(f"{url}|{chunk_id}".encode()).hexdigest()


def main() -> None:
    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY in .env")
    if not DATA_FILE.exists():
        raise FileNotFoundError("Run: python crawl_allergy.py")

    documents = load_documents(DATA_FILE)
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    records = []
    for chunk in chunks:
        vid = vector_id(chunk.metadata.get("url", ""), int(chunk.metadata["chunk_id"]))
        chunk.metadata["vector_id"] = vid
        records.append({"id": vid, "text": chunk.page_content, "metadata": chunk.metadata})
    save_json(records, CHUNKS_FILE)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    names = {x.name for x in pc.list_indexes()}
    if PINECONE_INDEX not in names:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    description = pc.describe_index(PINECONE_INDEX)
    if int(description.dimension) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Index dimension {description.dimension} does not match embedding dimension "
            f"{EMBEDDING_DIMENSION}. Create a new {EMBEDDING_DIMENSION}-dimension index."
        )

    index = pc.Index(PINECONE_INDEX)
    # index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
    try:
        index.delete(
            delete_all=True,
            namespace=PINECONE_NAMESPACE
        )
        print(f"Cleared existing namespace: {PINECONE_NAMESPACE}")

    except Exception as e:
        if "Namespace not found" in str(e):
            print(
                f"Namespace '{PINECONE_NAMESPACE}' does not exist yet. "
                "It will be created automatically during upload."
            )
        else:
            raise
    store = PineconeVectorStore(index=index, embedding=get_embeddings(EMBEDDING_MODEL), namespace=PINECONE_NAMESPACE)
    store.add_documents(chunks, ids=[x["id"] for x in records])
    print(f"Indexed {len(chunks)} chunks into {PINECONE_INDEX}/{PINECONE_NAMESPACE}")


if __name__ == "__main__":
    main()
