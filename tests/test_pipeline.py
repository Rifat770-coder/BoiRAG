import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from openai import RateLimitError
from streamlit.testing.v1 import AppTest

from config import DIMENSION, EVALUATION, FALLBACK, ROOT, read_json
from evaluation.evaluate import evaluate_run, hit_rate, retrieval_row
from src.crawler import ORIGIN, crawl
from src.preprocessing import make_chunks
from src.rag import Answer, BookRAG, invoke_with_backoff
from src.vector_store import load_index, retriever_for, save_index


BOOK = ORIGIN + "/wiki/Book"
API_REQUEST = httpx.Request("POST", "https://example.test")
FIRST = BOOK + "/First"
PARENT = BOOK + "/Section"
SECOND = PARENT + "/Second"
PASSAGE = "সে নদীর ধারে বসে ছিল।"


def verification_response(answer):
    return json.dumps({"supported": bool(answer), "answer": answer})


def page(title, content):
    return f'<h1 id="firstHeading">{title}</h1><div class="mw-parser-output">{content}</div>'


def link(url):
    return f'<a href="{url}">অধ্যায়</a>'


class FixtureEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        first = "নদী" in text
        return [float(first), float(not first)] + [0.0] * (DIMENSION - 2)


class PipelineTests(unittest.TestCase):
    def test_rate_limit_backoff_and_provider_retry_hint(self):
        response = httpx.Response(429, headers={"Retry-After": "6"}, request=API_REQUEST)
        error = RateLimitError("Please retry in 7s.", response=response, body=None)
        model = Mock()
        model.invoke.side_effect = [error, error, "completed"]
        with patch("src.rag.sleep") as pause, patch("src.rag.uniform", return_value=0):
            self.assertEqual(invoke_with_backoff(model, "prompt"), "completed")
        self.assertEqual([call.args[0] for call in pause.call_args_list], [7, 8])
        self.assertEqual(model.invoke.call_count, 3)

    def test_rate_limit_stops_at_budget_or_exhausted_quota(self):
        for wait, message, call_count in [
            ("1", "Rate limited", 3),
            ("120", "Rate limited", 1),
            ("1", "GenerateRequestsPerDay", 1),
            ("1", "Quota exceeded, limit: 0", 1),
        ]:
            with self.subTest(wait=wait, message=message):
                response = httpx.Response(429, headers={"Retry-After": wait}, request=API_REQUEST)
                model = Mock()
                model.invoke.side_effect = RateLimitError(message, response=response, body=None)
                with patch("src.rag.sleep") as pause, self.assertRaises(RateLimitError):
                    invoke_with_backoff(model, "prompt")
                self.assertEqual(model.invoke.call_count, call_count)
                self.assertEqual(pause.call_count, call_count - 1)

    def crawl_fixture(self, missing=False):
        html = {
            BOOK: '<div class="mw-parser-output">Status badge</div>'
            + '<div id="mw-content-text">'
            + page("বই", link(FIRST) + link(PARENT) + link(FIRST) + link(ORIGIN + "/wiki/Other"))
            + "</div>",
            FIRST: page("প্রথম", f'<div class="ws-footer">navigation</div><p>{PASSAGE}</p>'),
            PARENT: page("বিভাগ", link(SECOND) + link(FIRST)),
            SECOND: page("দ্বিতীয়", "<p>তারা গ্রামের দিকে গেল।</p>"),
        }

        def get(url, **kwargs):
            response = Mock(is_redirect=False, content=html[url].encode("utf-8"))
            if missing and url == SECOND:
                from requests import HTTPError

                response.raise_for_status.side_effect = HTTPError("404")
            return response

        with patch("src.crawler.requests.Session") as session:
            session.return_value.get.side_effect = get
            return crawl(BOOK, "Test fixtures", [FIRST, SECOND])

    def test_crawl_nested_order_duplicate_and_utf8(self):
        chapters, report = self.crawl_fixture()
        self.assertTrue(report["complete"])
        self.assertEqual([row["source_url"] for row in chapters], [FIRST, SECOND])
        self.assertEqual(chapters[0]["text"], PASSAGE)
        self.assertEqual(report["duplicate_links"], [FIRST])
        self.assertEqual(report["failed_urls"], {})

    def test_failed_chapter_prevents_completion(self):
        _, report = self.crawl_fixture(missing=True)
        self.assertFalse(report["complete"])
        self.assertIn(SECOND, report["failed_urls"])
        self.assertEqual(report["missing_chapters"], [SECOND])

    def test_chunk_metadata_and_faiss_roundtrip(self):
        chapters, _ = self.crawl_fixture()
        chunks = make_chunks(chapters, 800, 120)
        self.assertEqual([doc.metadata["chunk_id"] for doc in chunks], ["1-1", "2-1"])
        self.assertEqual(chunks[0].page_content, PASSAGE)
        with TemporaryDirectory() as directory, patch("src.vector_store.INDEXES", Path(directory)):
            embeddings = FixtureEmbeddings()
            save_index("baseline", chunks, embeddings, "fixture-corpus")
            store = load_index("baseline", embeddings, "fixture-corpus")
            self.assertEqual(store.index.d, DIMENSION)
            self.assertEqual(retriever_for(store).invoke("নদী কোথায়?")[0].metadata, chunks[0].metadata)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_index("baseline", embeddings, "different-corpus")
            binary = Path(directory) / "baseline" / "index.faiss"
            binary.write_bytes(binary.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                load_index("baseline", embeddings, "fixture-corpus")

    def test_quoted_support_and_abstention(self):
        doc = Document(page_content=PASSAGE, metadata={"chunk_id": "1-1", "source_url": FIRST})
        draft = '{"answer":"গ্রামে।","evidence":[{"chunk_id":"1-1","quote":"' + PASSAGE + '"}]}'
        for quote, verdict, expected in [(PASSAGE, True, False), (PASSAGE, False, True),
                                           ("মনগড়া", True, True)]:
            replies = Mock(side_effect=[draft.replace(PASSAGE, quote),
                                        verification_response("নদীর ধারে।" if verdict else ""),
                                        verification_response("নদীর ধারে।")])
            wider = Mock()
            wider.invoke.return_value = [doc]
            rag = BookRAG(None, RunnableLambda(replies), fallback_retriever=wider)
            answer = rag.answer("সে কোথায় বসেছিল?", docs=[doc])
            self.assertEqual(answer.abstained, expected)
            if expected:
                self.assertEqual(answer.text, FALLBACK)
                self.assertEqual(answer.sources, [])
            else:
                self.assertEqual(answer.text, "নদীর ধারে")
                self.assertEqual(answer.sources[0]["source_url"], FIRST)
                wider.invoke.assert_not_called()
                self.assertEqual(replies.call_count, 3)

    def test_hit_requires_quote_and_excludes_out_of_scope(self):
        doc = Document(page_content=PASSAGE, metadata={"chunk_id": "1-1", "source_url": FIRST})
        case = dict(question="প্রশ্ন", answerable=True, source_url=FIRST, reference_passage=PASSAGE)
        hit = retrieval_row(case, [doc])
        miss = retrieval_row({**case, "reference_passage": "অন্য অংশ"}, [doc])
        excluded = retrieval_row({**case, "answerable": False}, [doc])
        self.assertEqual(hit["reference_rank"], 1)
        self.assertFalse(miss["hit"])
        self.assertIsNone(excluded["hit"])
        self.assertEqual(hit_rate([hit, miss, excluded]), 50)

    def test_answer_retries_with_additional_passages_after_abstaining(self):
        distractor = Document(page_content="অন্য ঘটনা", metadata={"chunk_id": "2-1"})
        evidence = Document(page_content=PASSAGE, metadata={"chunk_id": "1-1", "source_url": FIRST})
        first = Mock()
        first.invoke.return_value = [distractor]
        wider = Mock()
        wider.invoke.return_value = [distractor, evidence]
        replies = iter([
            '{"answer":"","evidence":[]}',
            '{"answer":"নদীর ধারে।","evidence":[{"chunk_id":"1-1","quote":"' + PASSAGE + '"}]}',
            verification_response("নদীর ধারে।"),
            verification_response("নদীর ধারে।"),
        ])
        rag = BookRAG(first, RunnableLambda(lambda _: next(replies)), fallback_retriever=wider)
        answer = rag.answer("সে কোথায় বসেছিল?")
        self.assertFalse(answer.abstained)
        self.assertEqual(answer.sources[0]["source_url"], FIRST)
        wider.invoke.assert_called_once_with(*first.invoke.call_args.args)

    def test_related_object_is_rejected_and_wider_citation_is_used(self):
        ring = "রহিম আয়েশাকে একটি আংটি দিলেন।"
        dress = "রহিম রিনাকে একটি পোশাক দিলেন।"
        question = "রহিম কাকে পোশাক দিয়েছিলেন?"
        chapters = [
            dict(book_title="বই", chapter_title=title, chapter_number=number,
                 source_url=url, text=quote)
            for number, (title, url, quote) in enumerate([
                ("প্রথম", FIRST, ring), ("দ্বিতীয়", SECOND, dress),
            ], 1)
        ]
        first_doc, wider_doc = [
            Document(page_content=chapter["text"],
                     metadata={**{k: v for k, v in chapter.items() if k != "text"}, "chunk_id": str(number)})
            for number, chapter in enumerate(chapters, 1)
        ]
        first = Mock()
        first.invoke.return_value = [first_doc]
        wider = Mock()
        wider.invoke.return_value = [first_doc, wider_doc]
        replies = Mock(side_effect=[
            json.dumps({"answer": "আয়েশাকে", "evidence": [{"chunk_id": "1", "quote": ring}]}),
            json.dumps({"supported": False, "answer": "আয়েশাকে"}),
            json.dumps({"answer": "রিনাকে", "evidence": [{"chunk_id": "2", "quote": dress}]}),
            verification_response("রিনাকে"),
            verification_response("রিনাকে"),
        ])
        rag = BookRAG(first, RunnableLambda(replies), fallback_retriever=wider, chapters=chapters)
        answer = rag.answer(question)
        self.assertEqual(answer.attempts, ["verification_rejected", "verified"])
        self.assertEqual(answer.text, "রিনাকে")
        self.assertEqual([source["source_url"] for source in answer.sources], [SECOND])
        wider.invoke.assert_called_once()

        wider_doc.metadata["source_url"] = FIRST
        rag = BookRAG(first, RunnableLambda(lambda _: json.dumps({
            "answer": "রিনাকে", "evidence": [{"chunk_id": "2", "quote": dress}],
        })), chapters=chapters)
        self.assertEqual(rag.answer(question, docs=[wider_doc]).reason, "invalid_chapter")

    def test_cutting_bonds_does_not_answer_later_observation(self):
        case = read_json(EVALUATION / "test_questions.json")[5]
        cutting = ("এই বলিয়া কপালকুণ্ডলা অতি শীঘ্র হস্তে নবকুমারের লতাবন্ধন "
                   "খড়্গ দ্বারা ছেদন করিতে লাগিলেন। নিমেষ মধ্যে তাহাকে মুক্ত করিলেন।")
        observed = case["reference_passage"]
        first = Document(page_content=cutting, metadata={"chunk_id": "action"})
        second = Document(page_content=observed,
                          metadata={"chunk_id": "observation", "source_url": case["source_url"]})
        wider = Mock()
        wider.invoke.return_value = [first, second]
        replies = iter([
            json.dumps({"answer": cutting, "evidence": [{"chunk_id": "action", "quote": cutting}]}),
            verification_response(""),
            json.dumps({"answer": case["expected_answer"],
                        "evidence": [{"chunk_id": "observation", "quote": observed}]}),
            verification_response("ছিন্ন লতা বন্ধন"),
            verification_response("ছিন্ন লতা বন্ধনের উপর দৃষ্টি পড়িল"),
        ])
        rag = BookRAG(None, RunnableLambda(lambda _: next(replies)), fallback_retriever=wider)
        answer = rag.answer(case["question"], docs=[first])
        wider.invoke.assert_called_once()
        self.assertEqual(answer.attempts, ["verification_rejected", "verified"])
        self.assertEqual(answer.text, "ছিন্ন লতা বন্ধনের উপর দৃষ্টি পড়িল")
        self.assertEqual([source["source_url"] for source in answer.sources], [case["source_url"]])

    def test_blended_location_fails_span_check_and_recovers(self):
        case = read_json(EVALUATION / "test_questions.json")[8]
        passage = (case["reference_passage"]
                   + " তাহারই সম্মুখে আরও বৃহত্তর দ্বিতীয় এক খণ্ড সিকতাময় স্থান। সেই সৈকতে শ্মশানভূমি।")
        first = Document(page_content=passage,
                         metadata={"chunk_id": "sites", "source_url": case["source_url"]})
        wider = Mock()
        wider.invoke.return_value = [first, Document(page_content="অন্যান্য তথ্য",
                                                     metadata={"chunk_id": "extra"})]
        draft = json.dumps({"answer": case["expected_answer"],
                            "evidence": [{"chunk_id": "sites", "quote": passage}]})
        replies = iter([
            draft, verification_response("গঙ্গাতীরে"),
            verification_response("গঙ্গাতীরের এক বৃহৎ সৈকতভূমিতে অবস্থিত শ্মশানভূমিতে"),
            draft, verification_response("গঙ্গাতীরে"),
            verification_response("গঙ্গাতীরে এক বৃহৎ সৈকতভূমি"),
        ])
        rag = BookRAG(None, RunnableLambda(lambda _: next(replies)), fallback_retriever=wider)
        answer = rag.answer(case["question"], docs=[first])
        wider.invoke.assert_called_once()
        self.assertEqual(answer.attempts, ["unsupported_answer_span", "verified"])
        self.assertEqual(answer.text, "গঙ্গাতীরে এক বৃহৎ সৈকতভূমি")

    def test_evaluator_keeps_answer_and_citation_judgments_independent(self):
        case = read_json(EVALUATION / "test_questions.json")[8]
        quote = case["reference_passage"]
        chapter = dict(book_title="বই", chapter_title=case["expected_chapter"],
                       chapter_number=32, source_url=case["source_url"], text=quote)
        source = {**chapter, "quote": quote, "chunk_id": "site"}
        rag = Mock()
        rag.answer.return_value = Answer("শ্মশানভূমি", [source], False)
        grade_answer = Mock()
        grade_answer.invoke.return_value = {"answer_correct": False}
        grade_citations = Mock()
        grade_citations.invoke.return_value = {"citations_relevant": True}
        row = {"answerable": True, "hit": True}
        with patch("evaluation.evaluate.write_json") as save:
            passed, interrupted = evaluate_run(
                "baseline", None, rag, grade_answer, grade_citations,
                [(case, [], row, {"site": 1})], {case["source_url"]: chapter}, "fixture",
            )
        self.assertFalse(passed)
        self.assertIsNone(interrupted)
        answer_payload = json.loads(grade_answer.invoke.call_args.args[0]["case"])
        citation_payload = json.loads(grade_citations.invoke.call_args.args[0]["case"])
        self.assertNotIn("quotes", answer_payload)
        self.assertNotIn("expected_answer", citation_payload)
        self.assertNotIn("reference_passage", citation_payload)
        result = save.call_args.args[1]["cases"][0]
        self.assertIs(result["answer_correct"], False)
        self.assertIs(result["citations_relevant"], True)
        self.assertFalse(result["automated_pass"])

    def test_streamlit_session_displays_answer_and_chapter(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)
        answer = Answer("নদীর ধারে।", [{
            "chapter_title": "প্রথম", "source_url": FIRST, "quote": PASSAGE,
        }], False)
        store = Mock()
        store.as_retriever.return_value.invoke.return_value = []
        with (
            patch("src.vector_store.verified_corpus", return_value=(
                [{"book_title": "বই", "source_url": FIRST}], "ui"
            )),
            patch("src.vector_store.load_index", return_value=store),
            patch("src.embeddings.load_embeddings"),
            patch("src.rag.load_llm", return_value=RunnableLambda(lambda _: "")),
            patch.object(BookRAG, "answer", return_value=answer),
        ):
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.caption[0].value, "বই")
            app.chat_input[0].set_value("সে কোথায় বসেছিল?").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message), 2)
            self.assertEqual(app.expander[0].label, "প্রথম")
            self.assertIn(PASSAGE, [item.value for item in app.text])
            app.run()
            self.assertEqual(len(app.chat_message), 2)

    def test_streamlit_missing_book_shows_setup_message(self):
        with patch("src.vector_store.BOOK_URL", ""):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.info), 1)
        self.assertIn("BOOK_URL", app.markdown[0].value)
