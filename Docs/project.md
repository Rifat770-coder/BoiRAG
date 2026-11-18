# MASTER PROMPT — Bengali Wikisource Book RAG Chatbot

## ROLE

You are a Senior Python Developer, RAG Engineer, and LangChain Specialist.

Your task is to build a **complete, working, submission-ready Bengali Book Knowledge Base Chatbot using a Vector Database**, strictly following my university assignment.

**Primary objective:** Satisfy every requirement in the 100-mark evaluation rubric, plus the 10-mark bonus experiment.

Do not add unnecessary features, unrelated integrations, advanced agent systems, authentication, dashboards, or functionality outside the assignment.

Implement real, executable code. Do not create a mockup, placeholder application, fake benchmark, or hardcoded chatbot responses.

---

## 1. ASSIGNMENT RESOURCES

Carefully read these two Google Colab notebooks before implementation.

**Resource 01 — Bengali FAQ RAG Chatbot**

https://colab.research.google.com/drive/164hjcy-WNXFTnl3JJvMTaZd3DJ0lvYRn?usp=sharing

Learn from:

- Bengali sentence embeddings
- LangChain Document objects
- FAISS vector stores
- Similarity search
- Metadata handling
- Context-grounded LLM answers

**Resource 02 — RAG with LangChain, FAISS and Ollama**

https://colab.research.google.com/drive/1z-QM_9964Wiv8i94Nk6Hdtpmk4SekNFN?usp=sharing

Learn from:

- Web document loading
- RecursiveCharacterTextSplitter
- Embedding generation
- FAISS.from_documents()
- LangChain retrievers
- ChatPromptTemplate
- RAG chain construction
- Returning answers with retrieved documents

Use these notebooks as learning references, not as code to copy blindly.

If notebook links are inaccessible, clearly state the limitation and request the notebook `.ipynb` files. Do not claim that you reviewed inaccessible content.

**Important differences from the notebooks:**

- Use the selected Bengali Wikisource book instead of manually written FAQ data.
- Crawl every chapter subpage.
- Use a Bengali-compatible multilingual embedding model.
- Preserve chapter-level source metadata.
- Implement citations and reliable no-answer handling.
- Do not rebuild the vector index for every question.
- Implement actual hit-rate evaluation for the bonus.

---

## 2. ASSIGNMENT SCOPE

Build a chatbot that answers questions about exactly ONE completed Bengali prose book from:

https://bn.wikisource.org/

Selected book URL:

`[PASTE_SELECTED_BOOK_URL_HERE]`

The book must be:

- A completed prose book.
- Not poetry.
- Available as clean text without OCR.
- Registered by the student in the assignment's shared sheet.
- Crawled from all chapter subpages, not only the main page.

Never invent missing chapters, book passages, answers, or source URLs.

If the selected book URL is missing, do not invent a book selection. Complete the project structure and configurable implementation, and explicitly identify the book URL as a required input before ingestion and evaluation.

---

## 3. REQUIRED TECHNOLOGY

Use:

- Python
- Requests and BeautifulSoup for Wikisource crawling
- LangChain for document processing and retrieval
- RecursiveCharacterTextSplitter
- `BAAI/bge-m3` multilingual embedding model
- FAISS vector database
- A configurable Bangla-capable LLM through an API
- Streamlit for the chat interface
- Pandas or standard Python for evaluation
- Git and GitHub

Use compatible, maintained package versions and imports. Verify current LangChain package paths before implementation.

The application must run locally.

Store the API key securely in environment variables. Never hardcode credentials.

---

## 4. IMPLEMENTATION REQUIREMENTS

### TASK 01 — Full Book Ingestion (15 Marks)

Create a Wikisource crawler.

Requirements:

1. Accept the selected Bengali Wikisource book URL.
2. Fetch the book's main page.
3. Discover all chapter or section subpage links.
4. Follow nested chapter subpages where applicable.
5. Avoid duplicate pages, circular links, and unrelated links.
6. Extract the full readable content of every chapter.
7. Preserve the correct chapter order.
8. Remove navigation menus, footers, and unwanted markup.
9. Preserve the original Bengali text.
10. Save the extracted chapters as structured JSON.

Each chapter must contain:

- Book title
- Chapter title
- Chapter number or order
- Original chapter URL
- Full chapter text

Generate an ingestion report showing:

- Total discovered chapters
- Successfully extracted chapters
- Failed chapter URLs
- Total extracted characters
- Duplicate detection
- Empty chapter detection

Compare the extracted chapter manifest against the book's chapter listing.

If any chapter is missing, report the failure explicitly rather than pretending ingestion is complete.

### TASK 02 — Chunking and Preprocessing (15 Marks)

Preprocess the extracted book text.

Requirements:

- Normalize Unicode and unnecessary whitespace conservatively.
- Preserve Bengali punctuation, dialogue, names, and meaning.
- Avoid deleting important literary content.
- Split each chapter into chunks using LangChain.
- Never combine two different chapters into a single chunk.

Use these baseline settings:

`chunk_size = 800`

`chunk_overlap = 120`

Use Bengali-aware separators, including paragraph breaks and sentence punctuation.

Attach metadata to every chunk:

- book_title
- chapter_title
- chapter_number
- source_url
- chunk_id

Display total chunk count and sample chunks with metadata.

Document why the chosen chunk size and overlap are reasonable. Treat them as initial choices, not proven optimal settings.

### TASK 03 — Multilingual Embeddings (20 Marks)

Implement embeddings using:

`BAAI/bge-m3`

Requirements:

1. Load the multilingual model.
2. Generate embeddings for every chunk.
3. Use the same embedding model for user questions.
4. Confirm embedding vector dimensionality.
5. Handle model-loading and embedding errors.
6. Document why a multilingual embedding model is necessary for Bengali text.
7. Include the model name, link, and technical justification in README.

Do not use an English-only default embedding model for the final application.

### TASK 04 — FAISS + LangChain Retriever (15 Marks)

Create a FAISS vector store from the embedded chunks.

Requirements:

- Use LangChain's FAISS integration.
- Build the vector index once during ingestion/indexing.
- Save the index locally.
- Reload the saved index when starting the chatbot.
- Do not regenerate embeddings on every question.
- Convert the vector store to a LangChain retriever.
- Use `k=5` for initial retrieval.
- Preserve document metadata throughout retrieval.

Use safe FAISS loading practices and only load trusted locally generated index files.

Verify retrieval by printing the matched chapter titles and relevant text excerpts.

### TASK 05 — RAG Answers, Citations and No-Answer (20 Marks)

Create the complete RAG question-answering pipeline.

Flow:

User Question → Embedding → FAISS Retrieval → Relevant Context → LLM → Answer + Source Citations

Requirements:

1. Accept Bengali questions.
2. Retrieve relevant passages using LangChain.
3. Generate answers using only retrieved book passages.
4. Respond naturally in Bengali.
5. Cite the actual chapter or section.
6. Provide the original Wikisource source URL.
7. Show a supporting passage for each cited answer.
8. Never fabricate facts, citations, chapter names, or source links.
9. Do not use the LLM's general knowledge as evidence.
10. If the retrieved context cannot support an answer, abstain.

Use this exact fallback response:

"দুঃখিত, এই প্রশ্নের উত্তর নির্বাচিত বইয়ের তথ্যের মধ্যে পাওয়া যায়নি।"

Implement actual evidence sufficiency checking, not prompt instructions alone.

Do not treat a low similarity score as a universal proof of no answer. Evaluate retrieval confidence and contextual support using an approach calibrated against test questions.

Citations must be derived from the metadata of retrieved source documents.

Ensure the cited passage genuinely supports the generated answer.

Use a low generation temperature for consistency.

### TASK 06 — Streamlit Chat Interface (5 Marks)

Build a clean, functional Streamlit interface.

It must contain:

- Application title
- Selected book name
- Chat conversation area
- Bengali question input
- Generated answer
- Chapter citation
- Clickable Wikisource source link
- Supporting source passage
- Chat history during the session

Use Streamlit's native chat components.

Show appropriate loading indicators and friendly error messages.

Keep the UI simple and professional.

Do not add authentication, payments, user accounts, unrelated dashboards, or unnecessary animations.

### TASK 07 — Code and README (5 Marks)

Create a clean GitHub-ready repository.

Include:

- Complete Python source code
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `README.md`
- Selected book URL
- Installation instructions
- Run instructions
- Embedding model details and justification
- Chunk size and overlap explanation
- FAISS and LangChain explanation
- Brief architecture description
- Testing instructions
- Bonus experiment results
- Known limitations, if any

Use modular code with reasonable error handling.

No hardcoded API keys or secrets.

Include commands for:

1. Installing dependencies
2. Configuring environment variables
3. Running book ingestion
4. Building the FAISS index
5. Launching the Streamlit application
6. Running evaluation

### TASK 08 — Ten Test Questions (5 Marks)

Create exactly 10 mandatory test questions based on the selected book.

Each test case must contain:

- Question
- Expected answer
- Expected chapter
- Source URL
- Whether the question is answerable from the book

Include different types of questions:

- Direct factual questions
- Character-related questions
- Story or event-related questions
- Questions requiring understanding of passages
- At least one out-of-scope question

All expected answers must be verified against the actual source text.

Do not invent expected answers.

Save the test questions in a reusable structured file.

Run the tests and report actual results.

---

## 5. MANDATORY BONUS EXPERIMENT (+10 Marks)

The assignment offers 10 bonus marks for comparing two chunking strategies or embedding models using a simple hit-rate score.

Implement a real comparison between TWO chunking strategies.

**Strategy A**

- Chunk size: 500 characters
- Chunk overlap: 75 characters

**Strategy B**

- Chunk size: 1000 characters
- Chunk overlap: 150 characters

Keep everything else identical:

- Same book
- Same preprocessing rules
- Same BGE-M3 embedding model
- Same retrieval algorithm
- Same Top-K value (`k=5`)
- Same evaluation questions

Create separate FAISS indexes for Strategy A and Strategy B.

Use a fixed, manually source-verified set of answerable evaluation questions.

The mandatory 10 test questions must remain part of the submission. You may additionally create a small benchmark dataset for more reliable retrieval comparison.

Each benchmark question must have a verified reference passage or supporting source location.

Calculate:

`Hit Rate@5 = (Number of answerable questions with at least one relevant passage in the top 5 retrieved chunks / Total number of answerable questions) × 100`

A hit must require an actually relevant supporting passage, not merely a matching chapter title.

For questions that are unanswerable from the book, evaluate no-answer behaviour separately. Do not count those questions as retrieval misses in the Hit Rate@5 denominator.

Create an evaluation script that:

1. Loads both FAISS indexes.
2. Executes identical questions against both retrievers.
3. Checks retrieved passages against verified reference evidence.
4. Records hit or miss for each answerable question.
5. Computes Hit Rate@5 for each strategy.
6. Saves actual results to CSV or JSON.
7. Produces a small comparison table or chart.
8. Explains which strategy performs better based on measured results.

Do not fabricate benchmark numbers.

Do not claim that one strategy is better before executing the experiment.

Select the better-performing strategy for the final application where appropriate, and document the decision.

---

## 6. REQUIRED PROJECT STRUCTURE

Use a simple repository organization:

```text
bangla-book-rag/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── src/
│   ├── crawler.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── rag.py
├── scripts/
│   ├── ingest.py
│   └── build_index.py
├── evaluation/
│   ├── test_questions.json
│   ├── evaluate.py
│   └── compare_chunking.py
├── data/
└── indexes/
```

Additional files are allowed only when required for a working implementation.

---

## 7. VERIFICATION REQUIREMENTS

Before declaring completion, verify every marking criterion.

Check:

- Every chapter was crawled successfully.
- Extracted text matches the selected book.
- Bengali text is preserved.
- Metadata survives chunking and retrieval.
- Embeddings are generated using the selected multilingual model.
- FAISS index persists and reloads.
- LangChain retrieval works.
- Answers are grounded in retrieved passages.
- Source citations are correct.
- No-answer questions do not produce fabricated responses.
- Streamlit runs successfully.
- Ten expected-answer test cases are included.
- Bonus comparison is actually executed.
- Hit-rate results are measured, not estimated.
- README reproduces the application setup.

For every criterion, report:

- Implementation file
- Verification method
- Actual test result
- Pass / Fail / Not Tested

Do not claim full marks or successful tests without supporting evidence.

Fix discovered errors and rerun affected tests.

---

## 8. DEMO VIDEO REQUIREMENTS

Prepare a recording script for a 3–5 minute demonstration.

The script must show:

1. The selected Bengali Wikisource book.
2. Full-book chapter ingestion.
3. Chunking and embedding.
4. FAISS vector store and retriever.
5. Running Streamlit chatbot.
6. Exactly five demonstrated questions.
7. At least one demonstrated question with no answer in the book.
8. Chapter citations and original source links.
9. Bonus experiment results.
10. GitHub repository and README.

Prepare the script, but do not fabricate a recorded video.

---

## 9. DEVELOPMENT WORKFLOW

Complete the project in this order:

Phase 1 — Validate the selected book and chapter structure.

Phase 2 — Implement and test complete-book crawling.

Phase 3 — Implement preprocessing and baseline chunking.

Phase 4 — Generate multilingual embeddings and build the FAISS index.

Phase 5 — Implement the LangChain RAG pipeline with citations and no-answer handling.

Phase 6 — Build and test the Streamlit chat interface.

Phase 7 — Create and execute the 10 mandatory test cases.

Phase 8 — Implement and run the bonus chunking comparison.

Phase 9 — Complete README, dependencies, and demo script.

Phase 10 — Perform final verification against the 100-mark rubric and 10-mark bonus.

After each phase, check that the relevant code works before continuing.

Keep a concise progress report identifying completed, failed, and pending tasks.

Do not stop after making a frontend mockup or a project skeleton.

Where execution is unavailable due to missing credentials, hardware, source access, or environment restrictions, finish all feasible implementation and clearly identify what remains untested. Never invent successful execution.

---

## 10. FINAL OUTPUT

Deliver:

1. Fully implemented source code.
2. Working Bengali Wikisource crawler.
3. Preprocessing and chunking pipeline.
4. Multilingual embeddings.
5. Persistent FAISS database.
6. LangChain RAG chatbot.
7. Streamlit interface.
8. Ten verified test questions with expected answers.
9. Real bonus hit-rate comparison.
10. README and requirements.txt.
11. Demo video recording script.
12. Final marks-criteria verification table.

**Final instruction:** Strictly prioritize the assignment's eight marking criteria and the bonus experiment. Do not waste time on unrelated features. Implement, test, verify, and document all required functionality before declaring the assignment complete.