import argparse
import json

from config import DIMENSION, MODEL, STRATEGIES
from src.embeddings import load_embeddings
from src.preprocessing import make_chunks
from src.vector_store import load_index, retriever_for, save_index, verified_corpus


def main():
    parser = argparse.ArgumentParser(description="Build or verify a local FAISS index.")
    parser.add_argument("--strategy", choices=STRATEGIES, default="baseline")
    parser.add_argument("--query", help="Verify retrieval using a saved index without rebuilding it.")
    args = parser.parse_args()
    chapters, corpus_hash = verified_corpus()
    embeddings = load_embeddings()
    if args.query:
        store = load_index(args.strategy, embeddings, corpus_hash)
        for doc in retriever_for(store).invoke(args.query):
            print(f"{doc.metadata['chapter_title']} | {doc.metadata['source_url']}")
            print(doc.page_content[:240])
        return
    chunks = make_chunks(chapters, *STRATEGIES[args.strategy])
    print(f"{args.strategy}: {len(chunks)} chunks | {MODEL} | {DIMENSION} dimensions")
    for doc in chunks[:3]:
        print(json.dumps(doc.metadata, ensure_ascii=False))
        print(doc.page_content[:240])
    save_index(args.strategy, chunks, embeddings, corpus_hash)
    load_index(args.strategy, embeddings, corpus_hash)
    print("Saved and reloaded the local FAISS index.")


if __name__ == "__main__":
    main()
