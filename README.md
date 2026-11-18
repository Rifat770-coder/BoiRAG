# BoiRAG

![BoiRAG dashboard showing the Kapalkundala book sidebar, suggested Bengali questions, and the response to an out-of-scope greeting](Docs/Dashboard.png)

BoiRAG answers Bengali questions about one prose book from Bengali Wikisource. It cites chapter URLs and shows supporting Bengali text for accepted answers.

The dashboard shows the active book, its 32 chapters, the baseline index's 276 chunks, and the BGE-M3 embedding model. Readers can type a question, pick a suggested question, or browse chapter sources from the sidebar. The screenshot shows how the app responds when a greeting falls outside the book's scope.

The selected novel is **কপালকুণ্ডলা** by বঙ্কিমচন্দ্র চট্টোপাধ্যায়. The supplied [short link](https://w.wiki/jqMM) points to the 1867 scan index. This project ingests the [1870 chapter-based transcription](https://bn.wikisource.org/wiki/%E0%A6%95%E0%A6%AA%E0%A6%BE%E0%A6%B2%E0%A6%95%E0%A7%81%E0%A6%A3%E0%A7%8D%E0%A6%A1%E0%A6%B2%E0%A6%BE_(%E0%A6%AC%E0%A6%99%E0%A7%8D%E0%A6%95%E0%A6%BF%E0%A6%AE%E0%A6%9A%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0_%E0%A6%9A%E0%A6%9F%E0%A7%8D%E0%A6%9F%E0%A7%8B%E0%A6%AA%E0%A6%BE%E0%A6%A7%E0%A7%8D%E0%A6%AF%E0%A6%BE%E0%A6%AF%E0%A6%BC%2C_%E0%A7%A7%E0%A7%AE%E0%A7%AD%E0%A7%A6)), with all 32 chapter URLs in the [ordered manifest](Docs/kapalkundala_chapters.json). The two Colab notebooks linked in [the assignment](Docs/project.md) could not be opened.

## Setup

Use 64-bit Python 3.12 or later with compatible FAISS and PyTorch wheels. From the repository root on PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set these values in `.env`.

- `BOOK_URL` holds the canonical Bengali Wikisource article for the single completed, registered prose book. Confirm that the source is readable and that the listing covers the full book.
- `WIKISOURCE_USER_AGENT` identifies the application and includes your contact information under the [Wikimedia User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy).
- `CHAPTER_MANIFEST` points to an ordered UTF-8 JSON array such as `data/chapter_manifest.json`. Set it after auditing the book listing as described below.
- `LLM_API_KEY` and `LLM_MODEL` configure a Bengali-capable API model that supports Chat Completions, JSON object response format, and temperature zero.
- `LLM_BASE_URL` optionally selects another OpenAI-compatible provider endpoint. The provider receives the question and retrieved passages when generating an answer.
- `EMBEDDING_DEVICE` defaults to `cpu`. A supported CUDA environment can use `cuda`.
- `EMBEDDING_MODEL_PATH` optionally points to a local copy of the official BGE-M3 files.
- `INDEX_STRATEGY` defaults to `baseline`. Choose `a` or `b` after comparing retrieval and checking answer quality.

Keep `.env` out of Git. The first BGE-M3 run downloads the model weights and requires adequate disk space and memory.

For Gemini, use your Google API key as `LLM_API_KEY`, set `LLM_MODEL=gemini-3.1-flash-lite`, and set `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`. [Google Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)

## Ingest and run

Once the book index and `.env` are ready, double-click [start.bat](start.bat) or run it from the repository root:

```powershell
.\start.bat
```

In Git Bash, use `cmd.exe //c start.bat`. The launcher uses the project's `.venv` and starts the app at `http://127.0.0.1:8501`. To use a different port, run `.\start.bat --server.port 8502`.

Run discovery first with `CHAPTER_MANIFEST` unset.

```powershell
.\.venv\Scripts\python.exe -m scripts.ingest
```

The first run writes `data/ingestion_report.json` and `data/chapters.json` and exits with status 1 because completeness awaits a manual listing check. Compare discovered pages against the actual table of contents and nested listings. Record **every text-bearing chapter URL in book order** in your manifest. Include a parent section if it has prose of its own, and explicitly include any chapters outside the book's URL prefix. Check that extracted text matches the individual articles. Rerun ingestion with `CHAPTER_MANIFEST` set, or pass `--manifest data/chapter_manifest.json`.

The report records discovered subpages, expected and extracted chapter order, duplicate links and duplicate text, empty and failed pages, unmatched links, and extracted character count. Only a complete report for the selected book unlocks indexing. Redirected pages require their canonical URLs. Unusual Wikisource layouts may require selector adjustments after source inspection.

```powershell
.\.venv\Scripts\python.exe -m scripts.build_index --strategy baseline
.\.venv\Scripts\python.exe -m scripts.build_index --strategy baseline --query "বইটির মূল ঘটনা কী?"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Index creation prints the chunk count, configured dimensions, and sample metadata and text. The retrieval command prints titles, URLs, and excerpts from saved vectors. When rebuilding an index for the same corpus, restart Streamlit to reload it.

## Pipeline

The Streamlit entry point is `app.py`. The `ui/` directory holds presentation code and assets, `src/` holds the crawler and RAG pipeline, and `scripts/` holds the ingestion and indexing commands. The `evaluation/` directory contains test questions and benchmark commands. The app reads its configuration from `.env` and uses the generated corpus and FAISS files under `data/` and `indexes/`.

1. Requests fetches the landing page and chapter pages. Beautiful Soup discovers nested subpages and extracts readable text in chapter order.
2. NFC normalization and whitespace cleanup preserve Bengali punctuation. LangChain splits chapters individually with paragraph and Bengali sentence boundaries. The baseline uses 800 characters and a 120-character target overlap to fit nearby dialogue and narrative context within a retrieval unit. Recursive splitting can yield a smaller actual overlap at natural boundaries. These numbers are starting settings.
3. [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) provides 1,024-dimensional dense multilingual vectors for the Bengali passages and questions in the same semantic space. The same normalized embedding configuration is used at build and query time. Its multilingual training makes it appropriate for Bengali retrieval, with effectiveness measured by the assignment benchmark.
4. LangChain FAISS uses squared L2 distance on normalized vectors. The chatbot starts with five passages. If it cannot verify an answer, it searches up to 40 results and keeps the original passages available alongside the additional results. Vectors persist as a native FAISS file, with documents and metadata in JSON. The loader checks corpus and index hashes and avoids pickle deserialization. Only load files generated locally with the build command. Hashes detect mismatches and do not authenticate files from other people.
5. The model proposes verbatim evidence. The application checks the quote against its retrieved chunk and the chapter metadata against the saved corpus. A relevance check looks for the exact event and relationship in the question. A separate extraction step keeps essential details such as a cause or the terrain of the requested site. Both steps must return a span found in the authenticated quote. A rejected verdict or ungrounded span triggers wider retrieval. Insufficient support after that pass returns the assignment's exact Bengali fallback.

HTTP 429 can trigger up to two retries with exponential backoff and jitter. The retry waits honor longer provider hints within a 60-second per-wait budget. A hint beyond that budget or an explicit exhausted daily or zero quota stops the request. The evaluation command can space draft, verifier, grader and retry requests with `--request-interval`. Set the interval to suit the project's current quota. [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)

## Ten required questions and bonus

[evaluation/test_questions.json](evaluation/test_questions.json) contains nine questions with verified source passages and one out-of-scope question. For a different book, supply exactly ten distinct questions spanning direct facts, characters, events, and passage comprehension, including at least one out-of-scope question. Each record has these keys.

| Key | Value |
| --- | --- |
| `question` | Bengali question |
| `expected_answer` | Manually verified answer, or `দুঃখিত, এই প্রশ্নের উত্তর নির্বাচিত বইয়ের তথ্যের মধ্যে পাওয়া যায়নি।` |
| `expected_chapter` | Exact extracted chapter title, or JSON null for out-of-scope |
| `source_url` | Exact chapter URL from ingestion, or JSON null for out-of-scope |
| `answerable` | JSON true or false |
| `reference_passage` | Short continuous source quote sufficient to support the answer, or JSON null for out-of-scope |

Review the quote in the original source and keep it within 500 characters. The scripts verify literal containment in the corpus. A human must verify that the quote actually supports the expected answer.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m evaluation.evaluate --strategy baseline --request-interval 9 --runs 3
.\.venv\Scripts\python.exe -m scripts.build_index --strategy a
.\.venv\Scripts\python.exe -m scripts.build_index --strategy b
.\.venv\Scripts\python.exe -m evaluation.compare_chunking
.\.venv\Scripts\python.exe -m evaluation.evaluate --strategy a
.\.venv\Scripts\python.exe -m evaluation.evaluate --strategy b
```

The bonus uses the same source corpus, embeddings, retriever and questions at 500/75 and 1000/150 characters. Hit Rate@5 is the percent of answerable questions where a top-five chunk from the verified source contains the supporting quote. A quote split across chunk boundaries counts as a miss under this metric. The comparison prints a Markdown table and saves its results and winning strategy or tie. Out-of-scope abstention is measured separately by the end-to-end evaluator.

The measured results on the nine answerable questions were:

| Strategy | Size / overlap | Chunks | Hits | Hit Rate@5 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 800 / 120 | 276 | 5/9 | 55.56% |
| A | 500 / 75 | 440 | 5/9 | 55.56% |
| B | 1000 / 150 | 218 | 4/9 | 44.44% |

Strategy A won the bonus comparison. The app uses the baseline, which matched A's hit rate with 164 fewer chunks. The regenerated comparison and its per-case ranks are saved locally in `evaluation/results/chunking_comparison.json`.

Before the current grounding changes, three paced Gemini 3.1 Flash-Lite evaluations each scored 9/10 automatically and 8/10 on source review. In the latest completed development trial, source review passed 9/10. Cases 6 and 9 were correct, and case 8 regressed to the ring answer. The final implementation combines the relevance check that recovered case 8 in an earlier trial with the complete-span extraction that recovered cases 6 and 9. Gemini interrupted its first live run with HTTP 429 after generating the answer to case 1. The final three-run score is unavailable.

The evaluator records the initial and wide reference ranks, ranks of cited chunks, answer correctness, citation authenticity and relevance, and the exact no-answer check. Answer grading receives the verified reference and generated answer. Citation grading receives the question, generated answer, and actual quotes in a separate call. A repeated evaluation saves a separate `evaluation-baseline-run-N.json` for each completed run. A provider interruption writes a `-partial.json` report and stops the batch. Pending cases have a null grade.

See the [verification report](Docs/verification.md) and the [demo recording script](Docs/demo.md).
