import argparse
import hashlib
import json
import os
from pathlib import Path

from config import BOOK_URL, DATA, read_json, write_json
from src.crawler import crawl


def main():
    parser = argparse.ArgumentParser(description="Crawl and audit the selected Bengali book.")
    parser.add_argument("--manifest", default=os.getenv("CHAPTER_MANIFEST", ""))
    args = parser.parse_args()
    if not BOOK_URL:
        parser.error("Set BOOK_URL to the registered Bengali Wikisource book.")
    expected = read_json(Path(args.manifest)) if args.manifest else None
    chapters, report = crawl(BOOK_URL, os.getenv("WIKISOURCE_USER_AGENT", ""), expected)
    corpus = DATA / "chapters.json"
    write_json(corpus, chapters)
    report["corpus_sha256"] = hashlib.sha256(corpus.read_bytes()).hexdigest()
    write_json(DATA / "ingestion_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
