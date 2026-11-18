# Demo recording script

Target length is 4 minutes and 30 seconds. Prepare the verified book, baseline and bonus indexes, and ten completed test cases before recording. Choose four verified answerable questions and one verified out-of-scope question from that file.

| Time | Screen and narration |
| --- | --- |
| 0:00–0:35 | Open the selected Bengali Wikisource book. Name it, show its table of contents, and open a chapter subpage. |
| 0:35–1:05 | Show the ingestion command and report from the completed run. Compare the ordered manifest to the listing and show the chapter count, failure count, and a sample of the extracted Bengali text. |
| 1:05–1:40 | Show the baseline build command and actual output. Point out 800/120 chunking, the BGE-M3 embedding dimension and metadata, and the local FAISS and JSON files. Show the retrieval output with chapter titles and excerpts. |
| 1:40–3:30 | Open the running Streamlit app. Ask exactly five questions in the order below. Open at least one citation and follow its Wikisource link. |
| 3:30–4:05 | Show the measured comparison table for 500/75 and 1000/150, explain what qualified as a hit, and identify the measured winner or tie. Mention the separate abstention result. |
| 4:05–4:30 | Show the GitHub repository and README, including setup, selected book URL, and measured results. |

1. Ask a direct factual question and compare the answer with its displayed supporting quote.
2. Ask a character question and expand the citation.
3. Ask about an event and follow the chapter link to the original source.
4. Ask a passage comprehension question and explain how the quote supports the response.
5. Ask the out-of-scope question and show the Bengali fallback.
