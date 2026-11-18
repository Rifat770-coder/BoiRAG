# Assignment verification

The repository was checked on Windows with CPython 3.14.7. The supplied URL opens a scan index for the 1867 edition of কপালকুণ্ডলা. The corpus uses Wikisource's readable 1870 transcription of the same novel. The two assignment notebooks were inaccessible and shared-sheet registration could not be confirmed from the available materials.

The eight required criteria total 100 marks. The chunking comparison is worth up to 10 bonus marks. The table records verification evidence for grading.

| Criterion | Rubric marks | Implementation | Verification method and actual result | Status |
| --- | ---: | --- | --- | --- |
| Full book ingestion | 15 | [Crawler](../src/crawler.py), [ingestion command](../scripts/ingest.py) | A fresh Wikisource crawl matched the checked-in manifest and saved corpus exactly in contents order. It extracted 32 chapters and 151,340 characters, with zero missing, failed, unexpected, empty, or duplicate-text chapters. | Passed on Wikisource. |
| Chunking and preprocessing | 15 | [Preprocessing](../src/preprocessing.py) | The book produced 276 baseline, 440 strategy A, and 218 strategy B chunks. All nine reference passages remained intact in at least one chunk for each strategy before retrieval. | Passed on book text. |
| Multilingual embeddings | 20 | [Embeddings](../src/embeddings.py) | The local BGE-M3 weight file matched the official Hugging Face LFS SHA-256. The 276 saved baseline vectors and fresh Bengali queries have 1,024 dimensions. | Passed on book text. |
| FAISS and LangChain retrieval | 15 | [Vector store](../src/vector_store.py), [index command](../scripts/build_index.py) | The 276-vector baseline reloaded with chapter metadata. Exact reference passages appeared in 5/9 initial searches and 9/9 searches at rank 40 or better. The fixture rejected altered index bytes and changed corpus hashes. | Retrieval passed. Four references require wider retrieval. |
| Grounded answers and abstention | 20 | [RAG](../src/rag.py) | Source review confirmed 7/9 answerable cases and 1/1 out-of-scope cases in each of three live runs. All 27 citations matched the corpus and chunks. Case 8 recovered at rank 12 in each run. | Partial. Cases 6 and 9 failed semantic grounding in all three runs. |
| Streamlit chat | 5 | [Interface](../app.py) | The local server passed its health check. Streamlit AppTest opened the real index and displayed chat input with Gemini configured. A fixture test checked an answer, chapter passage, and conversation history. | Configured startup and fixture chat passed. |
| Code and README | 5 | [Setup and usage](../README.md), [dependencies](../requirements.txt), [environment example](../.env.example) | Dependency compatibility, Python compilation, CLI help, and all 11 local tests passed. Tests cover rejection of unrelated evidence, wider recovery, wrong chapter metadata, top-five short-circuiting, and bounded 429 retry. | Passed local checks. |
| Ten mandatory test questions | 5 | [Questions](../evaluation/test_questions.json), [evaluator](../evaluation/evaluate.py) | Three paced live runs each completed all ten cases. Each scored 9/10 automatically and 8/10 on source review. | Dataset and execution passed. Two answers remain incorrect. |
| Bonus comparison | +10 | [Comparison](../evaluation/compare_chunking.py) | Built and reloaded both BGE-M3 FAISS indexes. Strategy A hit 5/9 (55.56%), strategy B hit 4/9 (44.44%), and the baseline hit 5/9 (55.56%). The unanswerable case was excluded from retrieval scoring. | Passed with measured results. |
| Demo preparation | N/A | [Recording script](demo.md) | The script covers the source, pipeline, five questions including an abstention, measured results, and the repository in 4 minutes 30 seconds. No video was recorded. | Script ready. |

## Why the initial five cases failed

Case numbers follow the order in [test_questions.json](../evaluation/test_questions.json). The initial run is preserved locally in `evaluation/results/evaluation-baseline-initial.json`. The rank column is from a new search on the same baseline FAISS index and refers to the first chunk containing the complete reference passage in the expected chapter.

| Case | Reference rank | Saved outcome and diagnosis |
| --- | ---: | --- |
| 4. Two villages | 34 | Retrieval miss. The top five omitted the passage naming দৌলতপুর and দরিয়াপুর. The saved result shows insufficient evidence. |
| 5. Prohibition until the next meeting | 15 | Retrieval miss. The top five omitted both the reference about staying in the hut and the earlier chapter's direct dialogue. The saved result shows insufficient evidence. |
| 6. Sign of escape | 11 | Retrieval miss and verifier rejection. The exact passage about the severed vine bonds was below the cutoff. A draft with a locally matched quote reached the verifier, which rejected it. The initial log contains the rejection reason. The rejected draft and quote were discarded, so the semantic basis for the verdict is unknown. |
| 8. Recipient of the dress | 12 | Retrieval miss, off-topic generation and verifier false acceptance. The draft admitted it lacked the dress answer and substituted a ring given to কপালকুণ্ডলা in chapter 30. The exact dress passage identifies পেষ্‌মন্‌. The saved quote was authentic and described a different event. The then-current evaluator failed the answer at its expected-chapter gate. |
| 9. Place of worship | 1 | Generation error and verifier false acceptance. The cited paragraph places the worship site on a large sandbank by the Ganges. The cremation ground is described on the second sandy tract opposite it. The verifier accepted the conflation and the answer failed grading. |

The draft and verifier now require evidence for the exact fact asked and distinguish nearby events and locations. The recovery pass retains the top five passages alongside ranks 6–40 so an answer can use evidence across both groups. The first rerun recovered case 4 and answered case 5 correctly, then hit HTTP 429 on case 6. Case 5 had been marked wrong because its source was the first volume's fourth chapter, which contains the original dialogue. The benchmark's reference is in the next chapter, where the prohibition is recalled. The evaluator now checks every returned citation against the corpus and asks the grader to check whether the cited quote supports the expected fact. Hit Rate@5 still uses the original reference chapter and passage.

The first paced run completed all ten cases and passed 9/10. It recovered cases 4, 5, 6 and 8, and the out-of-scope case returned the exact fallback with no citations. Case 9 still gave the location of the cremation ground even though its quote contained the correct location. The draft and verifier were then tightened to resolve which location each sentence describes. The complete 9/10 run is preserved locally in `evaluation/results/evaluation-baseline-paced-first.json`.

The second paced run completed all ten cases and passed 7/10. Case 9 repeated the same location error. In cases 6 and 8, the draft denied that the requested fact was available, supplied a related fact, and received an affirmative verifier verdict. These three errors occurred before the wide pass because the verifier had accepted the drafts. The complete run is saved as `evaluation/results/evaluation-baseline-paced-second.json`.

The third paced run gave the verifier just the question and authenticated quotes. It passed 7/10. Cases 3 and 9 abstained after the verifier returned invalid response fields twice. Case 8 falsely accepted the ring passage again. The run is saved as `evaluation/results/evaluation-baseline-paced-third.json`. The verifier now returns a single answer field and explicitly checks that the quote identifies the requested object. The wide pass remains available whenever the first set of quotes cannot answer the question.

## Previous live evaluation

On 19 November 2026, the baseline run with `--request-interval 8` completed all ten cases and passed nine. Cases 4, 5, 6 and 9 recovered from the initial run. Case 9 correctly located the worship site on a large sandbank by the Ganges. Case 8 remained a false acceptance. The returned quote is from chapter 30 and names কপালকুণ্ডলা as the recipient of a ring. The dress is offered to পেষ্‌মন্‌ in chapter 20, with the relevant passage at rank 12. Accepting the first result prevented wider retrieval on that query. Case 8 passed in the first paced run, so the failure is intermittent.

All returned citation URLs, chapter titles, chapter numbers and quotes matched the freshly confirmed corpus and the saved chunks. The out-of-scope World Cup question returned the exact Bengali fallback and an empty citation list. Citation authenticity and support were checked separately. The unresolved ring citation is authentic and fails to support the dress answer.

The final [evaluation output](../evaluation/results/evaluation-baseline.json) has SHA-256 `6a092e4f5a0339b7171e3d52d0e5f8c41d91ab9a806380bf1c15ddd1753120d5`. Earlier outputs and the bonus comparison are saved locally under `evaluation/results/`.

## Evidence reliability regression

The application now checks the quoted text against the retrieved chunk and its chapter metadata against the verified corpus before calling the semantic verifier. The verifier compares participants, object, action and the requested relationship in the question and quote. Only an explicit positive verdict with an answer stops the search at five passages. A rejected or incomplete verdict allows recovery using the top 40. A supported first-pass answer still takes two model calls.

A live schema trial rejected the ring evidence and invoked the wider search, but strict extra output fields caused multiple false abstentions. That trial passed 2/10 and is preserved in `evaluation/results/evaluation-baseline-schema-trial.json`. The final verifier contract requires only the support verdict and answer.

## Three-run live verification

On 19 November 2026, three sequential Gemini 3.1 Flash-Lite runs used seven seconds between requests. All 30 cases completed. The verdict and citations were the same across the three runs.

| Run | Automated | Source review | Passed on source review | Failed on source review | Exact no-answer |
| --- | ---: | ---: | --- | --- | ---: |
| [1](../evaluation/results/evaluation-baseline-run-1.json) | 9/10 | 8/10 | 1–5, 7, 8, 10 | 6, 9 | 1/1 |
| [2](../evaluation/results/evaluation-baseline-run-2.json) | 9/10 | 8/10 | 1–5, 7, 8, 10 | 6, 9 | 1/1 |
| [3](../evaluation/results/evaluation-baseline-run-3.json) | 9/10 | 8/10 | 1–5, 7, 8, 10 | 6, 9 | 1/1 |

| Case | Exact reference rank | Cited rank | Citation authentic | Answer correct on source review | Citation supports the answer as stated |
| --- | ---: | ---: | --- | --- | --- |
| 1 | 1 | 1 | Yes | Yes | Yes |
| 2 | 1 | 1 | Yes | Yes | Yes |
| 3 | 4 | 4 | Yes | Yes | Yes |
| 4 | 34 | 34 | Yes | Yes | Yes |
| 5 | 15 | 31 | Yes | Yes | Yes |
| 6 | 11 | 2 | Yes | No | No |
| 7 | 1 | 1 | Yes | Yes | Yes |
| 8 | 12 | 12 | Yes | Yes | Yes |
| 9 | 1 | 1 | Yes | No | No |
| 10 | — | — | No citation | Exact fallback | — |

- **Case 6.** The first pass accepted chapter 6, which describes the act of cutting the bonds. The question asks what sign কাপালিক noticed afterward. The passage describing that discovery was available at rank 11, but the incorrect positive verdict stopped retrieval at five.
- **Case 8.** The verifier rejected the first-pass evidence in every run. Wider retrieval returned chapter 20 at rank 12, where লুৎফ্-উন্নিসা offers the dress to পেষ্‌মন্‌.
- **Case 9.** The rank-1 passage distinguishes the worship site's sandbank from the second tract across from it, which holds the cremation ground. The answer conflated those locations. Both the runtime verifier and automated grader accepted the answer. The source-review score counts it as a failure.

All 27 returned quotes and their chapter titles, numbers, and URLs were checked against both the saved chunks and the verified corpus. The three no-answer responses had no citations. The 100-mark rubric remains partially verified because semantic grounding failed in cases 6 and 9.

## Bonus evidence

The local [comparison output](../evaluation/results/chunking_comparison.json) was regenerated during this audit and checked against the active corpus, model ID, per-question ranks and winning strategy. The saved ranked chunk IDs for all three strategies were replayed against corpus SHA-256 `330c50531c90ebbfcf28850f0e62a122f34efacc03bfa40080cd3f1a5ee13d07`. Rank cells show where the exact reference appeared within five results. A dash indicates a miss.

| Case | Baseline | Strategy A | Strategy B |
| --- | ---: | ---: | ---: |
| 1 | 1 | 1 | 1 |
| 2 | 1 | 2 | — |
| 3 | 4 | 3 | 1 |
| 4 | — | — | — |
| 5 | — | — | — |
| 6 | — | — | — |
| 7 | 1 | 1 | 1 |
| 8 | — | — | — |
| 9 | 1 | 2 | 3 |
| Hit Rate@5 | 5/9 (55.56%) | 5/9 (55.56%) | 4/9 (44.44%) |
