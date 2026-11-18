from config import EVALUATION, MODEL, STRATEGIES, TOP_K, write_json
from evaluation.evaluate import hit_rate, load_questions, retrieval_row
from src.crawler import normalize
from src.embeddings import load_embeddings
from src.vector_store import load_index, retriever_for, verified_corpus


def main():
    chapters, corpus_hash = verified_corpus()
    cases = [case for case in load_questions(chapters) if case["answerable"]]
    embeddings = load_embeddings()
    results = {}
    print("| Strategy | Size | Overlap | Hits | Hit Rate@5 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for strategy in ("a", "b"):
        store = load_index(strategy, embeddings, corpus_hash)
        retriever = retriever_for(store)
        rows = [retrieval_row(case, retriever.invoke(normalize(case["question"]))) for case in cases]
        size, overlap = STRATEGIES[strategy]
        rate = hit_rate(rows)
        hits = sum(row["hit"] for row in rows)
        print(f"| {strategy.upper()} | {size} | {overlap} | {hits}/{len(rows)} | {rate:.2f}% |")
        results[strategy] = {"size": size, "overlap": overlap, "hits": hits,
                             "hit_rate_percent": rate, "cases": rows}
    a, b = (results[strategy]["hit_rate_percent"] for strategy in ("a", "b"))
    winner = None if a == b else ("a" if a > b else "b")
    conclusion = (
        f"Strategy {winner.upper()} retrieved verified evidence for more questions."
        if winner else "Both strategies retrieved verified evidence for the same number of questions."
    )
    print(conclusion)
    write_json(EVALUATION / "results" / "chunking_comparison.json", {
        "results": results, "winner": winner, "conclusion": conclusion,
        "embedding_model": MODEL, "k": TOP_K, "corpus_sha256": corpus_hash,
    })


if __name__ == "__main__":
    main()
