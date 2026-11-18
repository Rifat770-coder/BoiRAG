import os

import streamlit as st

from config import EVALUATION, FALLBACK_TOP_K, MODEL, read_json
from src.embeddings import load_embeddings
from src.rag import BookRAG, ModelNotConfigured, load_llm
from src.vector_store import BookNotConfigured, load_index, retriever_for, verified_corpus
from ui.presentation import book_details, chapter_label, render_header, render_sidebar, render_theme

st.set_page_config(page_title="বইয়ের সঙ্গে কথোপকথন", page_icon="📖", layout="wide")
render_theme()


@st.cache_resource
def get_rag(strategy, corpus_hash):
    llm = load_llm()
    store = load_index(strategy, load_embeddings(), corpus_hash)
    rag = BookRAG(retriever_for(store), llm, fallback_retriever=retriever_for(store, k=FALLBACK_TOP_K))
    return rag, store.index.ntotal


def render_answer(answer, message_index):
    st.text(answer.text)
    for source_index, source in enumerate(answer.sources):
        with st.container(key=f"citation-{message_index}-{source_index}"):
            with st.expander(chapter_label(source["chapter_title"]), expanded=True,
                             icon=":material/menu_book:"):
                st.text(source["quote"])
            st.link_button("অধ্যায়ে যান", source["source_url"], icon=":material/open_in_new:")


strategy = os.getenv("INDEX_STRATEGY", "baseline")
try:
    chapters, corpus_hash = verified_corpus()
    book_title = chapters[0]["book_title"]
    title, publication = book_details(book_title)
    render_header(title, publication, book_title, chapters)
    with st.spinner("বইয়ের সূচি লোড হচ্ছে…"):
        rag, chunk_count = get_rag(strategy, corpus_hash)
except BookNotConfigured:
    st.info("চ্যাট চালু করতে নিবন্ধিত বাংলা উইকিসংকলন বইয়ের ঠিকানা দিন।")
    st.markdown("`.env` ফাইলে `BOOK_URL` সেট করুন। এরপর README অনুযায়ী বই সংগ্রহ করে সূচি তৈরি করুন।")
    st.stop()
except ModelNotConfigured:
    st.info("চ্যাট চালু করতে `.env` ফাইলে `LLM_MODEL` ও `LLM_API_KEY` সেট করুন।")
    st.stop()
except Exception as exc:
    st.error("বইয়ের সূচি বা মডেল লোড করা যায়নি। সেটআপের জন্য README দেখুন।")
    if isinstance(exc, (ValueError, FileNotFoundError, RuntimeError)):
        st.text(str(exc))
    st.stop()

if st.session_state.get("corpus_hash") != corpus_hash:
    st.session_state.corpus_hash = corpus_hash
    st.session_state.messages = []

def clear_chat():
    st.session_state.messages = []


try:
    chapter_urls = {chapter.get("source_url") for chapter in chapters}
    test_questions = [
        case["question"] for case in read_json(EVALUATION / "test_questions.json")
        if case["answerable"] and case["source_url"] in chapter_urls
    ]
except (OSError, ValueError):
    test_questions = []

suggested = render_sidebar(title, publication, chapters, chunk_count, MODEL, test_questions, clear_chat)

for message_index, (question, answer) in enumerate(st.session_state.messages):
    with st.chat_message("user", avatar=":material/person:"):
        st.text(question)
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        render_answer(answer, message_index)

with st.bottom:
    if test_questions:
        with st.container(key="suggestions", horizontal=True, vertical_alignment="center"):
            st.markdown(":material/auto_awesome: **প্রস্তাবিত প্রশ্ন**")
            for index, prompt in enumerate(test_questions[:3]):
                if st.button(prompt, key=f"suggestion-{index}", icon=":material/menu_book:"):
                    suggested = prompt
    question = st.chat_input("বইটি সম্পর্কে বাংলায় প্রশ্ন করুন") or suggested

if not st.session_state.messages and not question:
    with st.container(key="welcome"):
        st.markdown("### বইটি সম্পর্কে কী জানতে চান?")

if question and question.strip():
    with st.chat_message("user", avatar=":material/person:"):
        st.text(question)
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        try:
            with st.spinner("বইয়ে প্রাসঙ্গিক অংশ খোঁজা হচ্ছে…"):
                answer = rag.answer(question)
        except Exception:
            st.error("উত্তর তৈরি করা যায়নি। মডেল সংযোগ পরীক্ষা করে আবার চেষ্টা করুন।")
        else:
            render_answer(answer, len(st.session_state.messages))
            st.session_state.messages.append((question, answer))
