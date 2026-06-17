"""
rag_engine.py
─────────────────────────────────────────────────────────────────────────────
Responsibilities:
  1. Parse a PDF file into raw text (page-by-page).
  2. Recursively split the text into overlapping chunks.
  3. Initialise the local HuggingFace sentence-transformer embedding model
     once and reuse it across calls (module-level singleton).
  4. Store document chunks + embeddings in an in-memory ChromaDB collection.
  5. Expose a retrieval function that returns the top-k most relevant chunks
     for a given query — used by quiz_generator.py at generation time.

Design goals
  • CPU-safe: no CUDA requirement; batch size kept small to limit RAM spikes.
  • Stateless between Streamlit re-runs: the collection is rebuilt whenever a
    new PDF is uploaded (call `ingest_pdf` again; the old collection is wiped).
  • Zero disk I/O for the vector store: EphemeralClient keeps everything in RAM.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import gc
import logging
import uuid
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # ~22 MB, CPU-friendly
COLLECTION_NAME      = "student_notes"
CHUNK_SIZE           = 512                   # characters per chunk
CHUNK_OVERLAP        = 64                    # overlap to preserve context
EMBED_BATCH_SIZE     = 16                    # small batches → lower peak RAM
TOP_K_DEFAULT        = 6                     # chunks returned per query

# ── Module-level singletons (initialised once per Python process) ──────────────
_embedding_model: SentenceTransformer | None = None
_chroma_client:   chromadb.EphemeralClient  | None = None
_collection:      chromadb.Collection       | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_embedding_model() -> SentenceTransformer:
    """
    Lazy-load the sentence-transformer model exactly once.
    `device='cpu'` is explicit so the model never tries to allocate GPU memory.
    """
    global _embedding_model
    if _embedding_model is None:
        log.info("Loading embedding model '%s' on CPU …", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu",
        )
        log.info("Embedding model loaded.")
    return _embedding_model


def _get_chroma_client() -> chromadb.EphemeralClient:
    """Return a module-level in-memory ChromaDB client (no disk writes)."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )
        log.info("ChromaDB EphemeralClient initialised.")
    return _chroma_client


def _reset_collection() -> chromadb.Collection:
    """
    Drop and recreate the collection so repeated PDF uploads start fresh.
    Returns the new, empty collection.
    """
    global _collection
    client = _get_chroma_client()

    # Delete if it already exists from a previous upload
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        log.info("Dropped existing ChromaDB collection '%s'.", COLLECTION_NAME)

    _collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine distance for sentence embeddings
    )
    log.info("Created fresh ChromaDB collection '%s'.", COLLECTION_NAME)
    return _collection


def _get_collection() -> chromadb.Collection:
    """Return the active collection (must call ingest_pdf first)."""
    if _collection is None:
        raise RuntimeError(
            "No ChromaDB collection found. Call `ingest_pdf()` before querying."
        )
    return _collection


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – PDF → raw text
# ══════════════════════════════════════════════════════════════════════════════

def parse_pdf(pdf_source: str | Path | bytes) -> str:
    """
    Extract and concatenate all text from a PDF.

    Args:
        pdf_source: A file path (str/Path) *or* raw bytes (from
                    Streamlit's `st.file_uploader`).

    Returns:
        A single string containing the full document text.
    """
    import io
    
    if isinstance(pdf_source, (str, Path)):
        # If it's a file path, open it explicitly in 'rb' (read binary) mode
        with open(str(pdf_source), "rb") as f:
            reader = PdfReader(f)
            pages = [page.extract_text() or "" for page in reader.pages]
    else:
        # If it's raw bytes from Streamlit, wrap it cleanly in BytesIO
        # Ensure we pass an open binary stream directly into PdfReader
        pdf_stream = io.BytesIO(pdf_source)
        reader = PdfReader(pdf_stream)
        pages = [page.extract_text() or "" for page in reader.pages]

    cleaned_pages: List[str] = []
    for i, page_text in enumerate(pages):
        # Prevent stray encoding characters from breaking the downstream loop
        text = page_text.encode("utf-8", errors="ignore").decode("utf-8")
        text = text.strip()
        if text:
            cleaned_pages.append(text)
        else:
            log.debug("Page %d yielded no extractable text — skipped.", i + 1)

    full_text = "\n\n".join(cleaned_pages)
    log.info("Parsed %d page(s), total length %d chars.", len(cleaned_pages), len(full_text))
    return full_text


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – text → chunks
# ══════════════════════════════════════════════════════════════════════════════

def split_text(text: str) -> List[str]:
    """
    Recursively split `text` into overlapping chunks.

    RecursiveCharacterTextSplitter tries to break on paragraph boundaries
    first, then sentences, then words — producing semantically coherent chunks
    rather than hard-cutting mid-sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
        length_function=len,
    )
    chunks = splitter.split_text(text)
    log.info("Split into %d chunks (size=%d, overlap=%d).",
             len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 & 4 – embed chunks + store in ChromaDB
# ══════════════════════════════════════════════════════════════════════════════

def _embed_in_batches(texts: List[str]) -> List[List[float]]:
    """
    Embed `texts` in small batches to keep peak RAM low on CPU.

    SentenceTransformer.encode() with `show_progress_bar=False` and a small
    `batch_size` avoids loading all activations at once.

    Returns:
        A list of embedding vectors (one per input text).
    """
    model = _get_embedding_model()
    all_embeddings: List[List[float]] = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        vecs = model.encode(
            batch,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,   # numpy → list conversion is fast
            normalize_embeddings=True,  # unit-norm; pairs well with cosine distance
        )
        all_embeddings.extend(vecs.tolist())

        # Encourage GC between batches to reclaim intermediate tensors
        gc.collect()

        log.debug(
            "Embedded batch %d–%d / %d",
            start + 1,
            min(start + EMBED_BATCH_SIZE, len(texts)),
            len(texts),
        )

    return all_embeddings


def store_chunks(chunks: List[str]) -> int:
    """
    Embed `chunks` and upsert them into a fresh ChromaDB collection.

    Returns:
        The number of chunks stored.
    """
    collection = _reset_collection()
    model      = _get_embedding_model()   # ensure model is warm before timing

    log.info("Embedding %d chunks …", len(chunks))
    embeddings = _embed_in_batches(chunks)

    # ChromaDB expects parallel lists: ids, embeddings, documents
    ids       = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    log.info("Stored %d chunks in ChromaDB.", len(chunks))
    return len(chunks)


# ══════════════════════════════════════════════════════════════════════════════
# Public API – ingest pipeline
# ══════════════════════════════════════════════════════════════════════════════

def ingest_pdf(pdf_source: str | Path | bytes) -> int:
    """
    Full pipeline: parse → split → embed → store.

    Call this once per uploaded PDF.  Calling it again with a new PDF
    automatically wipes the previous collection.

    Args:
        pdf_source: File path or raw bytes from Streamlit's file uploader.

    Returns:
        Number of chunks ingested.
    """
    raw_text = parse_pdf(pdf_source)
    if not raw_text.strip():
        raise ValueError(
            "The uploaded PDF contains no extractable text. "
            "Scanned/image-only PDFs are not supported without OCR."
        )

    chunks = split_text(raw_text)
    n      = store_chunks(chunks)
    return n


# ══════════════════════════════════════════════════════════════════════════════
# Public API – retrieval
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_context(query: str, top_k: int = TOP_K_DEFAULT) -> str:
    """
    Embed `query` and return the top-k most relevant chunks as a single string.

    This is the function called by quiz_generator.py to build the prompt
    context before sending it to the Groq API.

    Args:
        query:  A natural-language description of the topic(s) to quiz on.
        top_k:  Number of chunks to retrieve.

    Returns:
        Concatenated chunk texts separated by double newlines.
    """
    collection  = _get_collection()
    model       = _get_embedding_model()

    query_vec = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_vec,
        n_results=min(top_k, collection.count()),  # guard if fewer chunks exist
        include=["documents"],
    )

    docs: List[str] = results["documents"][0]   # list of chunk strings
    context = "\n\n".join(docs)
    log.info("Retrieved %d chunks for query: '%s …'", len(docs), query[:60])
    return context


# ══════════════════════════════════════════════════════════════════════════════
# Quick sanity-check (run directly: python rag_engine.py <path/to/test.pdf>)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_engine.py <path/to/notes.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"\n── Ingesting '{pdf_path}' ──")
    total = ingest_pdf(pdf_path)
    print(f"✓ Ingested {total} chunks.\n")

    test_query = "Explain the main concepts covered in these notes."
    print(f"── Sample retrieval for: '{test_query}' ──")
    ctx = retrieve_context(test_query, top_k=3)
    print(ctx[:800], "\n… (truncated)")