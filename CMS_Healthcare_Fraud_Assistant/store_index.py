from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX,
    PINECONE_NAMESPACE,
    PINECONE_REGION,
)
from src.helper import get_embeddings, load_json_documents, split_documents

DATA_FILE = Path("data/cms_documents.json")
CHUNKS_FILE = Path("data/cms_chunks.json")
BATCH_SIZE = 100


def require_settings() -> None:
    """Validate required settings and source files before indexing."""
    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY is missing. Add it to your .env file."
        )

    if not PINECONE_INDEX:
        raise ValueError(
            "PINECONE_INDEX is missing. Add it to your .env file."
        )

    if not PINECONE_NAMESPACE:
        raise ValueError(
            "PINECONE_NAMESPACE is missing. Add it to your .env file."
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "The CMS document file was not found. "
            "Run `python crawl_cms.py` before indexing."
        )


def stable_id(text: str, url: str, chunk_id: int) -> str:
    """Create a repeatable Pinecone vector ID for each text chunk."""
    raw_value = f"{url}|{chunk_id}|{text}"
    return hashlib.sha256(raw_value.encode("utf-8")).hexdigest()


def save_chunks(chunks, ids: list[str]) -> None:
    """Save a local copy of chunks for BM25/lexical retrieval."""
    records = []

    for chunk, vector_id in zip(chunks, ids):
        metadata = dict(chunk.metadata)
        metadata["vector_id"] = vector_id

        # Keep the vector ID in the in-memory document metadata too.
        chunk.metadata["vector_id"] = vector_id

        records.append(
            {
                "id": vector_id,
                "text": chunk.page_content,
                "metadata": metadata,
            }
        )

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_index_names(pc: Pinecone) -> set[str]:
    """Return existing Pinecone index names across SDK response formats."""
    response = pc.list_indexes()

    if hasattr(response, "names"):
        return set(response.names())

    names: set[str] = set()
    for item in response:
        if isinstance(item, dict):
            name = item.get("name")
        else:
            name = getattr(item, "name", None)

        if name:
            names.add(name)

    return names


def get_namespace_names(index) -> set[str]:
    """Return namespace names from describe_index_stats()."""
    stats = index.describe_index_stats()

    if isinstance(stats, dict):
        namespaces = stats.get("namespaces", {}) or {}
    else:
        namespaces = getattr(stats, "namespaces", {}) or {}

    if isinstance(namespaces, dict):
        return set(namespaces.keys())

    # Defensive fallback for unusual SDK response objects.
    try:
        return set(namespaces)
    except TypeError:
        return set()


def ensure_index(pc: Pinecone) -> None:
    """Create the Pinecone index when it does not already exist."""
    existing_names = get_index_names(pc)

    if PINECONE_INDEX in existing_names:
        print(f"Using existing Pinecone index: {PINECONE_INDEX}")
        return

    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
    )

    print(f"Created Pinecone index: {PINECONE_INDEX}")


def validate_index_dimension(pc: Pinecone) -> None:
    """Ensure the Pinecone index matches the embedding dimension."""
    description = pc.describe_index(PINECONE_INDEX)

    if isinstance(description, dict):
        dimension = description.get("dimension")
    else:
        dimension = getattr(description, "dimension", None)

    if dimension is None:
        raise ValueError(
            "Could not determine the Pinecone index dimension."
        )

    if int(dimension) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Pinecone index dimension is {dimension}, but the embedding "
            f"model uses {EMBEDDING_DIMENSION}. Delete and recreate the "
            "index with the correct dimension, or update EMBEDDING_DIMENSION."
        )


def clear_namespace_if_present(index) -> None:
    """Delete old vectors only when the configured namespace exists."""
    namespace_names = get_namespace_names(index)

    if PINECONE_NAMESPACE not in namespace_names:
        print(
            f"Namespace '{PINECONE_NAMESPACE}' does not exist yet. "
            "It will be created automatically during indexing."
        )
        return

    print(
        f"Deleting old vectors from namespace: {PINECONE_NAMESPACE}"
    )

    index.delete(
        delete_all=True,
        namespace=PINECONE_NAMESPACE,
    )

    print("Old namespace vectors deleted.")


def add_documents_in_batches(
    vector_store: PineconeVectorStore,
    chunks,
    ids: list[str],
    batch_size: int = BATCH_SIZE,
) -> None:
    """Upload documents in smaller batches for better reliability."""
    total = len(chunks)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        vector_store.add_documents(
            documents=chunks[start:end],
            ids=ids[start:end],
        )

        print(f"Indexed chunks {start + 1}-{end} of {total}")


def main() -> None:
    """Load CMS documents, split them, and index them in Pinecone."""
    require_settings()

    print(f"Loading CMS documents from: {DATA_FILE}")
    documents = load_json_documents(DATA_FILE)

    if not documents:
        raise ValueError(
            "No CMS documents were found in data/cms_documents.json. "
            "Run `python crawl_cms.py` again."
        )

    print(f"Loaded {len(documents)} CMS documents.")

    chunks = split_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        raise ValueError("No text chunks were created from the CMS documents.")

    print(f"Created {len(chunks)} text chunks.")

    ids = [
        stable_id(
            text=chunk.page_content,
            url=chunk.metadata.get("url", ""),
            chunk_id=int(chunk.metadata.get("chunk_id", 0)),
        )
        for chunk in chunks
    ]

    save_chunks(chunks, ids)
    print(f"Saved lexical-search chunks to: {CHUNKS_FILE}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = get_embeddings(EMBEDDING_MODEL)

    pc = Pinecone(api_key=PINECONE_API_KEY)

    ensure_index(pc)
    validate_index_dimension(pc)

    index = pc.Index(PINECONE_INDEX)

    print(f"Using namespace: {PINECONE_NAMESPACE}")
    clear_namespace_if_present(index)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )

    add_documents_in_batches(
        vector_store=vector_store,
        chunks=chunks,
        ids=ids,
    )

    print("\nIndexing completed successfully.")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Index: {PINECONE_INDEX}")
    print(f"Namespace: {PINECONE_NAMESPACE}")


if __name__ == "__main__":
    main()
