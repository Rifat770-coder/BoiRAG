import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
INDEXES = ROOT / "indexes"
EVALUATION = ROOT / "evaluation"
BOOK_URL = os.getenv("BOOK_URL", "").strip()
MODEL = "BAAI/bge-m3"
DIMENSION = 1024
TOP_K = 5
FALLBACK_TOP_K = 40
STRATEGIES = {"baseline": (800, 120), "a": (500, 75), "b": (1000, 150)}
FALLBACK = "দুঃখিত, এই প্রশ্নের উত্তর নির্বাচিত বইয়ের তথ্যের মধ্যে পাওয়া যায়নি।"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
