"""Streamlit rendering and display data for the literary interface."""

import base64
import re
from html import escape

import streamlit as st

from config import ROOT

ASSETS = ROOT / "ui" / "assets"
TITLE = "বইয়ের সঙ্গে কথোপকথন"


def asset_url(filename, mime):
    path = ASSETS / filename
    if not path.is_file():
        return ""
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def render_theme():
    font = asset_url("noto-serif-bengali.woff2", "font/woff2")
    landscape = asset_url("moonlit-river.webp", "image/webp")
    styles = (ASSETS.parent / "theme.css").read_text(encoding="utf-8")
    assets = f"""
    @font-face {{font-family: "Boi Bengali"; src:url("{font}") format("woff2");
                 font-weight:100 900; font-display:swap;}}
    .st-key-literary-header::before {{background-image:url("{landscape}");}}
    """
    st.html(f"<style>{styles}{assets}</style>")


def book_details(book_title):
    match = re.fullmatch(r"(.+?)\s*\(([^()]+)\)", book_title)
    return match.groups() if match else (book_title, "")


def chapter_label(title):
    parts = title.split("/")
    return " / ".join(parts[1:]) if len(parts) > 1 else title


def chapter_links(chapters):
    for chapter in chapters:
        if "source_url" in chapter and "chapter_title" in chapter:
            st.link_button(chapter_label(chapter["chapter_title"]), chapter["source_url"],
                           icon=":material/open_in_new:", width="stretch")


def render_header(title, publication, book_title, chapters):
    with st.container(key="literary-header"):
        st.html(f'<div class="literary-title"><span class="book-mark" aria-hidden="true"></span>'
                f'<h1>{TITLE}</h1></div>')
        st.caption(book_title)
        metadata = f'{title} — {publication}' if publication else title
        st.html(f'<p class="book-subtitle">{escape(metadata)}</p>')
        if chapters and "source_url" in chapters[0]:
            with st.popover("অধ্যায়সমূহ", icon=":material/search:", key="header-chapters"):
                with st.container(height=340, border=False):
                    chapter_links(chapters)


def render_sidebar(title, publication, chapters, chunk_count, model, prompts, clear_chat):
    digits = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
    with st.sidebar:
        st.html(f'<div class="sidebar-brand">{TITLE}</div>')
        cover = asset_url("kapalkundala.webp", "image/webp") if title == "কপালকুণ্ডলা" else ""
        author = publication.rsplit(",", 1)[0]
        cover_style = f' style="background-image:url({cover})"' if cover else ""
        st.html(f'<div class="book-cover"{cover_style}><strong>{escape(title)}</strong>'
                f'<span>{escape(author)}</span></div>'
                f'<div class="sidebar-book"><h2>{escape(title)}</h2>'
                f'<p>{escape(publication)}</p></div>')
        count = f"{chunk_count:,}".translate(digits) if isinstance(chunk_count, int) else "—"
        st.html(f"""<dl class="book-stats">
          <div><dt><span aria-hidden="true">▤</span> অধ্যায় (মোট)</dt>
            <dd>{str(len(chapters)).translate(digits)}</dd></div>
          <div><dt><span aria-hidden="true">◇</span> ইনডেক্সকৃত চাঙ্ক</dt><dd>{count}</dd></div>
          <div><dt><span aria-hidden="true">▣</span> অনুসন্ধান পদ্ধতি</dt>
            <dd><code title="{escape(model)}">{escape(model.split("/")[-1])}</code></dd></div>
        </dl>""")
        st.markdown(":material/bolt: **দ্রুত কাজ**")
        st.button("নতুন চ্যাট", type="primary", icon=":material/add:",
                  on_click=clear_chat, width="stretch")
        selected = None
        if prompts:
            with st.popover("টেস্ট প্রশ্ন", icon=":material/help:", width="stretch"):
                for index, prompt in enumerate(prompts):
                    if st.button(prompt, key=f"test-question-{index}", width="stretch"):
                        selected = prompt
        if chapters and "source_url" in chapters[0]:
            with st.popover("সূত্র দেখুন", icon=":material/menu_book:", width="stretch"):
                with st.container(height=340, border=False):
                    chapter_links(chapters)
    return selected
