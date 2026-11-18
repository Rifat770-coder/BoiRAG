"""Crawl Bengali Wikisource chapter links in the order they appear."""
import hashlib
import re
import unicodedata
from urllib.parse import quote, unquote, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ORIGIN = "https://bn.wikisource.org"
JUNK = (
    ".mw-editsection, .ws-noexport, .noprint, .navbox, .navigation-not-searchable, "
    ".licenseContainer, .catlinks, .mw-jump-link, #toc, "
    ".ws-header, .ws-footer, .prp-page-qualityheader, .pagenum, .reference"
)
NAVIGATION = ".ws-header, .ws-footer, .navbox, .navigation-not-searchable, .pagenum"


def normalize(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def article_url(value, base=ORIGIN):
    parsed = urlsplit(urljoin(base, value))
    if parsed.scheme != "https" or parsed.netloc != "bn.wikisource.org":
        raise ValueError("Use an HTTPS Bengali Wikisource article URL.")
    if not parsed.path.startswith("/wiki/") or parsed.query:
        raise ValueError("Use a Bengali Wikisource /wiki/ article URL without query parameters.")
    title = unicodedata.normalize("NFC", unquote(parsed.path[6:])).replace(" ", "_")
    if not title or ":" in title or any(part in {"", ".", ".."} for part in title.split("/")):
        raise ValueError("Use a main-namespace book or chapter article URL.")
    return f"{ORIGIN}/wiki/{quote(title, safe='/')}"


def parse_article(html, url, book_url):
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.select_one("#firstHeading, .mw-first-heading")
    body = soup.select_one("#mw-content-text > .mw-parser-output, #mw-content-text.mw-parser-output")
    if body is None:
        body = soup.select_one(".mw-parser-output")
    if heading is None or body is None:
        raise ValueError("Article heading or readable content is missing.")
    title = normalize(heading.get_text(" ", strip=True))
    for element in body.select(NAVIGATION):
        element.decompose()

    children = []
    for anchor in body.select("a[href]"):
        try:
            target = article_url(anchor["href"], url)
        except ValueError:
            continue
        if target.startswith(book_url + "/") and target != url:
            # Nested listings may link to siblings. Their parent owns their order.
            if url == book_url or target.startswith(url + "/"):
                children.append(target)

    for element in body.select(JUNK):
        element.decompose()
    for element in body.select("br"):
        element.replace_with("\n")
    for element in body.select("p, blockquote, li, dd, dt, h2, h3, h4, h5, tr, pre"):
        element.insert_before("\n\n")
        element.insert_after("\n\n")
    return title, normalize(body.get_text()), children


def crawl(book_url, user_agent, expected=None):
    book_url = article_url(book_url)
    if not user_agent.strip():
        raise ValueError("Set WIKISOURCE_USER_AGENT with your contact information.")
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    session.mount(ORIGIN, HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)
    )))

    if expected is not None:
        if not isinstance(expected, list) or not expected or not all(
            isinstance(url, str) for url in expected
        ):
            raise ValueError("The chapter manifest must be a nonempty ordered JSON array of URLs.")
        ordered = [article_url(url, book_url) for url in expected]
        if len(set(ordered)) != len(ordered) or book_url in ordered:
            raise ValueError("The chapter manifest contains a duplicate or the book landing page.")

    seen, traversal, duplicates, failures, empty, pages = set(), [], [], {}, [], {}

    def visit(url):
        if url in seen:
            duplicates.append(url)
            return
        seen.add(url)
        traversal.append(url)
        try:
            response = session.get(url, timeout=(10, 40), allow_redirects=False)
            response.raise_for_status()
            if response.is_redirect:
                raise ValueError("Redirect encountered. Use the canonical article URL.")
            title, content, children = parse_article(response.content.decode("utf-8"), url, book_url)
        except (requests.RequestException, ValueError) as exc:
            failures[url] = str(exc)
            return
        pages[url] = (title, content, children)
        for child in children:
            visit(child)

    with session:
        visit(book_url)
        discovered = [url for url in traversal if url != book_url]
        # An explicit listing can include substantive parent sections and off-tree chapters.
        if expected is not None:
            for url in ordered:
                if url not in seen:
                    visit(url)
            unexpected = [url for url in traversal if url != book_url and url not in ordered and
                          (url not in pages or not pages[url][2])]
        else:
            # Follow traversal order, excluding intermediate section tables of contents.
            ordered = [url for url in discovered if url not in pages or not pages[url][2]]
            unexpected = []

    book_title = pages.get(book_url, ("", "", []))[0]
    chapters, repeated_text, hashes = [], {}, {}
    for number, url in enumerate(ordered, 1):
        if url not in pages:
            continue
        title, content, _ = pages[url]
        if not content:
            empty.append(url)
            continue
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest in hashes:
            repeated_text[url] = hashes[digest]
        hashes.setdefault(digest, url)
        chapters.append(dict(book_title=book_title, chapter_title=title, chapter_number=number,
                             source_url=url, text=content))
    extracted = [chapter["source_url"] for chapter in chapters]
    missing = [url for url in ordered if url not in extracted]
    report = {
        "book_url": book_url, "book_title": book_title,
        "listing_source": "manual manifest" if expected is not None else "discovered leaf pages",
        "total_discovered_chapters": len(set(discovered) | set(ordered)),
        "successfully_extracted_chapters": len(chapters),
        "total_extracted_characters": sum(len(chapter["text"]) for chapter in chapters),
        "discovered_subpages": discovered, "expected_chapters": ordered,
        "extracted_chapters": extracted, "missing_chapters": missing,
        "unexpected_chapter_links": unexpected, "failed_urls": failures,
        "duplicate_links": duplicates, "duplicate_text": repeated_text, "empty_chapters": empty,
        "complete": bool(chapters and expected is not None and not (
            failures or missing or unexpected or repeated_text or empty
        )),
    }
    return chapters, report
