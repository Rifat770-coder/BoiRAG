# BoiRAG demo for Google Flow

Target runtime **3:00**. Record in landscape with English narration and the original Bengali chat and source passages on screen. Use Google Flow for the opening and closing visuals. The terminal and Streamlit footage supply the evidence that the pipeline runs.

## What the pipeline does

BoiRAG collects the readable 1870 Bengali Wikisource transcription of Bankim Chandra Chattopadhyay’s *কপালকুণ্ডলা*. It verifies the 32 chapters against an ordered manifest, normalizes the Bengali text, and splits each chapter into passages. The baseline uses 800 characters with a 120-character target overlap. BGE-M3 embeds the 276 passages into 1,024-dimensional vectors, which are saved in a local FAISS index with chapter metadata. When a question comes in, the app retrieves five passages and can widen the search to 40. The configured language model proposes evidence, and BoiRAG checks the quote against the indexed passage and verifies that it supports the requested fact. Streamlit shows the answer with the chapter passage and link, or returns the Bengali fallback when evidence is insufficient.

## Prepare the recording

Use the [README setup](../README.md) and the existing baseline configuration. Warm up the model and rehearse the five questions in a fresh chat. Check that each of the four answers is supported by its displayed quote and that the fifth reply has no citation. The latest evaluation attempt encountered HTTP 429. If the app shows a provider error, rerun the take after quota is available. A provider error is distinct from the book-only fallback. Keep `.env` and API keys outside the captured area.

Capture the commands below from the repository root. Use the checked-in ordered manifest. Record the start and successful end of slow commands, then trim the wait. Capture the retrieval output before starting Streamlit.

```powershell
.\.venv\Scripts\python.exe -m scripts.ingest --manifest Docs/kapalkundala_chapters.json
.\.venv\Scripts\python.exe -m scripts.build_index --strategy baseline
.\.venv\Scripts\python.exe -m scripts.build_index --strategy baseline --query "উপন্যাসের শুরুতে যাত্রীদের নৌকা কোথা থেকে ফিরছিল?"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Timeline and voiceover

| Time | Picture and action | Read aloud |
| --- | --- | --- |
| 0:00–0:05 | Flow opening clip with the title **BoiRAG · Ask a Bengali novel**. | “Ask a Bengali novel with BoiRAG.” |
| 0:05–0:16 | Show the 1870 Wikisource contents and open the first chapter. | “I’m using the Bengali Wikisource transcription of *Kapalkundala*. These chapter pages supply the text for every answer.” |
| 0:16–0:36 | Press Enter on the ingestion command. Show output appearing, then cut to the report confirming 32 chapters. | “The crawler collects the chapter text and checks it against the ordered manifest. This run ingested all 32 chapters.” |
| 0:36–0:59 | Run the baseline build. Cut to the reported 276 chunks and 1,024 dimensions. Run the saved-index query and show a returned Bengali excerpt. | “BGE-M3 embeds 276 passages into 1,024-dimensional vectors. FAISS saves those vectors locally. Here’s the saved index returning a chapter and excerpt for a Bengali question.” |
| 0:59–1:09 | Start Streamlit and cut to the loaded dashboard and chat input. | “Now I’ll ask five questions in the running app. Each supported answer shows its chapter evidence.” |
| 1:09–1:27 | Submit Q1 and hold on the live answer and citation. | “Where was the boat returning from? Gangasagar. The opening chapter names it directly.” |
| 1:27–1:45 | Submit Q2 and show the answer alongside the fog and navigation details. | “Why did the sailors lose the fleet? Thick fog kept them from determining their direction.” |
| 1:45–2:06 | Submit Q3, show the excerpt, click **অধ্যায়ে যান**, and return to chat. | “Why wouldn’t the passengers collect firewood? They feared tigers. I can open the cited chapter on Wikisource to check the passage.” |
| 2:06–2:24 | Submit Q4 and show the answer and excerpt from the second volume. | “What kind of jewelry box did Motibibi send? The quote describes an ivory box mounted with silver.” |
| 2:24–2:47 | Submit Q5. Hold on the live Bengali fallback and show that this reply has no chapter citation. | “Who won the 2022 FIFA World Cup? The novel doesn’t cover that. BoiRAG says it couldn’t find the answer in the selected book and gives no citation.” |
| 2:47–2:55 | Return to the app overview and briefly show the repository URL. | “That’s the full path from Wikisource to answers you can check against the book.” |
| 2:55–3:00 | Flow closing clip with **BoiRAG** and the repository URL. | |

For each question, keep the submission and completed reply visible. Trim the waiting time between them and hold the final response long enough to read. For Q5, capture the exact fallback from the app after the request completes successfully.

## Bengali questions to paste

The expected text below is a recording check. Keep the actual app output in the video. Cases 1, 2, 3, 7, and 10 are drawn from the [evaluation dataset](../evaluation/test_questions.json) and have supporting results in [previous verification](verification.md). They should still be rehearsed against the app you record.

| Shot | Paste into Streamlit | Expected evidence |
| --- | --- | --- |
| Q1 | উপন্যাসের শুরুতে যাত্রীদের নৌকা কোথা থেকে ফিরছিল? | গঙ্গাসাগর। প্রথম খণ্ড, প্রথম পরিচ্ছেদ। |
| Q2 | নাবিকেরা কী কারণে বহর থেকে বিচ্ছিন্ন হয়ে পড়েছিল? | ঘন কুয়াশায় দিক নির্ণয় করতে না পারায়। প্রথম খণ্ড, প্রথম পরিচ্ছেদ। |
| Q3 | রান্নার কাঠ জোগাড় করতে অন্য যাত্রীরা রাজি হননি কেন? | বাঘের ভয়ে। প্রথম খণ্ড, দ্বিতীয় পরিচ্ছেদ। |
| Q4 | মতিবিবি গহনা রাখার জন্য কী ধরনের কৌটা পাঠিয়েছিলেন? | রুপো দিয়ে বাঁধানো হাতির দাঁতের কৌটা। দ্বিতীয় খণ্ড, চতুর্থ পরিচ্ছেদ। |
| Q5 | ২০২২ সালের ফিফা বিশ্বকাপ কোন দেশ জিতেছিল? | দুঃখিত, এই প্রশ্নের উত্তর নির্বাচিত বইয়ের তথ্যের মধ্যে পাওয়া যায়নি। No citation. |

## Prompts to paste into Google Flow

**Opening**

```text
Landscape 16:9 cinematic shot for a Bengali literature software demo. An unmarked clothbound book lies open on a dark wooden desk beside a window. Beyond the window, low mist drifts across a Bengal river at night. Cool indigo moonlight and warm amber desk light, realistic paper texture, restrained movement, and a slow push toward the book. Leave dark open space on the left for a title added later. End with a steady frame for a clean cut. Silent. No lettering, logos, interface, or readable text.
```

**Closing**

```text
Landscape 16:9 closing shot matching the opening: the same unmarked clothbound book on the same dark wooden desk, cool indigo moonlight and warm amber desk light. Start close on the book and pull back slowly as mist moves over the Bengal river beyond the window. Hold the last frame with dark open space on the left for a title and link added later. Silent. No lettering, logos, interface, or readable text.
```

In Flow, create a project, choose Video and landscape, and generate one clip per prompt. Select a duration that can be trimmed to five seconds, and reuse a still from the opening as a visual reference for the closer if needed. Flow’s Scenebuilder can arrange, trim, preview, and download a sequence. For the full demo, bring the two exported clips into your editor alongside the original terminal and browser recordings. Lay in the voiceover, add the titles, and review the exported video at normal size for readable Bengali text, the five visible replies, and the chapter link. [Google Flow video generation](https://support.google.com/flow/answer/16353334) · [Google Flow Scenebuilder](https://support.google.com/flow/answer/16935718)
