from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.crawler import normalize


def make_chunks(chapters, size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "।", "॥", "?", "!", "…", ";", ",", " ", ""],
        keep_separator="end",
    )
    chunks = []
    for chapter in chapters:
        metadata = {key: value for key, value in chapter.items() if key != "text"}
        for number, content in enumerate(splitter.split_text(normalize(chapter["text"])), 1):
            chunk_id = f"{chapter['chapter_number']}-{number}"
            chunks.append(Document(page_content=content, metadata={**metadata, "chunk_id": chunk_id}))
    if not chunks:
        raise ValueError("The chapter corpus produced no chunks.")
    return chunks
