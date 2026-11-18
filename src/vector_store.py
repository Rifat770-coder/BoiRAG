"""Store FAISS vectors and JSON documents without unpickling Python objects."""
import hashlib
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import BOOK_URL, DATA, DIMENSION, INDEXES, MODEL, STRATEGIES, TOP_K, read_json, write_json
from src.crawler import article_url


class BookNotConfigured(ValueError):
    """The assignment's registered Wikisource book has not been selected."""


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verified_corpus():
    if not BOOK_URL:
        raise BookNotConfigured("Set BOOK_URL to the registered Bengali Wikisource book.")
    report = read_json(DATA / "ingestion_report.json")
    corpus_file = DATA / "chapters.json"
    corpus_hash = file_hash(corpus_file)
    if (not report["complete"] or report["book_url"] != article_url(BOOK_URL)
            or report["corpus_sha256"] != corpus_hash):
        raise ValueError("Run ingestion with a verified chapter manifest for the selected book.")
    return read_json(corpus_file), corpus_hash


def save_index(strategy, chunks, embeddings, corpus_hash):
    ids = [doc.metadata["chunk_id"] for doc in chunks]
    try:
        store = FAISS.from_documents(chunks, embeddings, ids=ids)
    except Exception as exc:
        raise RuntimeError("Embedding or FAISS index creation failed.") from exc
    if store.index.d != DIMENSION or store.index.ntotal != len(chunks):
        raise ValueError("FAISS index dimensions or record count are incorrect.")
    directory = INDEXES / strategy
    directory.mkdir(parents=True, exist_ok=True)
    # Stage files beside their destinations so they inherit the index directory's permissions.
    with ExitStack() as cleanup:
        staged = {}
        for filename in ("index.faiss", "index.json"):
            with NamedTemporaryFile(dir=directory, delete=False) as temporary:
                path = Path(temporary.name)
            cleanup.callback(path.unlink, missing_ok=True)
            staged[filename] = path
        vector_bytes = faiss.serialize_index(store.index).tobytes()
        staged["index.faiss"].write_bytes(vector_bytes)
        write_json(staged["index.json"], {
            "model": MODEL, "strategy": strategy, "settings": STRATEGIES[strategy],
            "corpus_sha256": corpus_hash, "index_sha256": hashlib.sha256(vector_bytes).hexdigest(),
            "documents": [doc.model_dump() for doc in chunks],
        })
        for filename, path in staged.items():
            path.replace(directory / filename)


def load_index(strategy, embeddings, corpus_hash):
    if strategy not in STRATEGIES:
        raise ValueError("INDEX_STRATEGY must be baseline, a, or b.")
    directory = INDEXES / strategy
    if directory.resolve() != INDEXES.resolve() / strategy:
        raise ValueError("Load only local indexes produced by the build command.")
    binary, metadata = directory / "index.faiss", directory / "index.json"
    if any(path.is_symlink() or path.resolve().parent != directory.resolve()
           for path in (binary, metadata)):
        raise ValueError("Linked index files are unsupported.")
    payload = read_json(metadata)
    if (payload["model"] != MODEL or payload["strategy"] != strategy
            or payload["settings"] != list(STRATEGIES[strategy])
            or payload["corpus_sha256"] != corpus_hash):
        raise ValueError("This index does not match the model, strategy, or ingested book. Rebuild it.")
    # Hash the exact bytes passed to FAISS. The digest detects mismatched local files.
    vector_bytes = binary.read_bytes()
    if hashlib.sha256(vector_bytes).hexdigest() != payload["index_sha256"]:
        raise ValueError("Index files are inconsistent. Rebuild the index.")
    docs = [Document.model_validate(doc) for doc in payload["documents"]]
    ids = [doc.metadata["chunk_id"] for doc in docs]
    index = faiss.deserialize_index(np.frombuffer(vector_bytes, dtype=np.uint8))
    if (not isinstance(index, faiss.IndexFlatL2) or index.d != DIMENSION
            or index.ntotal != len(ids) or len(ids) != len(set(ids))):
        raise ValueError("Local FAISS index is inconsistent. Rebuild it.")
    return FAISS(embeddings, index, InMemoryDocstore(dict(zip(ids, docs))), dict(enumerate(ids)))


def retriever_for(store, *, k=TOP_K):
    return store.as_retriever(search_kwargs={"k": k})
