#!/usr/bin/env python3
"""
Tests for batch processing and the book decision point.

The thing being protected is the *order of operations*. Classification is
cheap and reading is not, so every expensive decision has to be settled in the
pre-flight pass, before any GPU work — otherwise a folder of thirty PDFs with
one 345-page book in it becomes two silent hours nobody chose.

Three rules, each with a test that fails if it is broken:

  1. Nothing expensive happens without an explicit decision. `--noninteractive`
     always takes the CHEAP default; an unrecognised answer does too.
  2. A "chapter" of hundreds of pages is a misclassified book, and is refused
     rather than read — the `--chapters` opt-in guard covers the book path
     only, so this closes the gap on the chapter path.
  3. One bad PDF does not end a run.

The prompt is driven through injected `input_fn`/`output_fn`, so the
conversation is tested without a terminal — the same seam a GUI dialog will
replace.
"""

from pathlib import Path

import pytest

from src.processing.batch import (
    IMPLAUSIBLE_CHAPTER_PAGES,
    BookDecision,
    decision_for,
    PreflightEntry,
    find_pdfs,
    preflight,
    resolve_book_decisions,
    write_batch_summary,
)
from src.processing.book_prompt import (
    ask_about_book,
    parse_chapter_selection,
    show_preflight,
)
from src.processing.models import DocumentProfile, DocumentType
from src.processing.policy import BookPolicy, BookSettings


def profile(path, page_count, document_type, confidence=1.0):
    return DocumentProfile(pdf_path=path, page_count=page_count, file_size=10,
                           document_type=document_type,
                           classification_confidence=confidence)


def entry(document_type, page_count=300, name="book.pdf", **kwargs):
    return PreflightEntry(path=Path(name), document_type=document_type,
                          page_count=page_count, **kwargs)


class TestFindingPdfs:
    def test_pdfs_come_back_sorted_for_reproducible_runs(self, tmp_path):
        for name in ("c.pdf", "a.pdf", "b.pdf"):
            (tmp_path / name).write_bytes(b"%PDF")
        assert [p.name for p in find_pdfs(tmp_path)] == ["a.pdf", "b.pdf", "c.pdf"]

    def test_subdirectories_are_skipped_unless_asked_for(self, tmp_path):
        (tmp_path / "top.pdf").write_bytes(b"%PDF")
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "deep.pdf").write_bytes(b"%PDF")

        assert [p.name for p in find_pdfs(tmp_path)] == ["top.pdf"]
        assert len(find_pdfs(tmp_path, recursive=True)) == 2


class TestPreflight:
    def test_every_pdf_is_classified_before_anything_is_read(self, tmp_path):
        paths = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
        entries = preflight(paths, lambda path: profile(
            path, 8, DocumentType.RESEARCH_ARTICLE))

        assert [e.document_type for e in entries] == [DocumentType.RESEARCH_ARTICLE] * 2
        assert all(e.error is None for e in entries)

    def test_an_unreadable_pdf_is_recorded_not_raised(self, tmp_path):
        def explode(path):
            raise ValueError("Data format error")

        entries = preflight([tmp_path / "broken.pdf"], explode)
        assert "Data format error" in entries[0].error
        assert entries[0].document_type is None

    def test_a_book_is_described_so_the_prompt_shows_more_than_a_filename(self, tmp_path):
        entries = preflight(
            [tmp_path / "vol.pdf"],
            lambda path: profile(path, 345, DocumentType.BOOK),
            book_describer=lambda path: {"title": "Progress in Diseases",
                                         "chapter_count": 15})

        assert entries[0].title == "Progress in Diseases"
        assert entries[0].chapter_count == 15

    def test_a_failed_description_never_loses_the_entry(self, tmp_path):
        """Cosmetic detail must not cost a document."""
        def explode(path):
            raise RuntimeError("no contents page")

        entries = preflight([tmp_path / "vol.pdf"],
                            lambda path: profile(path, 345, DocumentType.BOOK),
                            book_describer=explode)

        assert entries[0].document_type is DocumentType.BOOK
        assert entries[0].page_count == 345


class TestTwoPdfsWithOneName:
    """Audit D3: outputs are named after the PDF, so two PDFs called
    paper.pdf in one batch used to overwrite each other's JSON, figures and
    tables without a word. The first runs; the second is held back."""

    @staticmethod
    def article(path):
        return profile(path, 8, DocumentType.RESEARCH_ARTICLE)

    def test_the_second_pdf_with_a_name_is_held_back_with_a_reason(self, tmp_path):
        first, second = tmp_path / "a" / "paper.pdf", tmp_path / "b" / "paper.pdf"
        classified = []
        entries = preflight([first, second],
                            lambda path: classified.append(path) or self.article(path))

        assert entries[0].error is None
        assert "same output name" in entries[1].error and str(first) in entries[1].error
        assert classified == [first], "a held-back PDF is not even opened"
        assert entries[1].held_back and "same name" in entries[1].describe()

    def test_names_that_differ_only_in_case_or_punctuation_also_clash(self, tmp_path):
        """Windows ignores case, and the figure folder turns spaces into _."""
        paths = [tmp_path / "a" / "My paper.pdf", tmp_path / "b" / "my_paper.PDF"]
        entries = preflight(paths, self.article)
        assert entries[0].error is None and entries[1].error is not None

    def test_different_names_never_clash(self, tmp_path):
        paths = [tmp_path / "a" / "paper.pdf", tmp_path / "a" / "paper2.pdf"]
        assert all(e.error is None for e in preflight(paths, self.article))

    def test_one_file_listed_twice_runs_once_without_an_error(self, tmp_path):
        """Results are keyed by path: a second entry would overwrite the
        first one's result with an error."""
        same = tmp_path / "paper.pdf"
        same.write_bytes(b"%PDF")
        entries = preflight([same, tmp_path / "." / "paper.pdf"], self.article)
        assert len(entries) == 1 and entries[0].error is None

    def test_a_held_back_pdf_is_listed_as_failed_in_the_summary(self, tmp_path):
        paths = [tmp_path / "a" / "paper.pdf", tmp_path / "b" / "paper.pdf"]
        entries = preflight(paths, self.article)
        summary = write_batch_summary(entries, {str(paths[0]): {"elapsed": 1.0}},
                                      tmp_path, 1.0)
        import json
        assert json.loads(summary.read_text(encoding="utf-8"))["failed"] == ["paper.pdf"]


def settings(policy="skip", **kwargs):
    return BookSettings(policy=BookPolicy.parse(policy), **kwargs)


class TestBookDecisions:
    """
    The policy says WHAT to do; the prompt handler says whether asking is
    possible. Keeping those separate is what lets the CLI and a GUI share one
    setting without either having to know the other exists.
    """

    def test_the_skip_policy_reads_no_bodies(self):
        entries = [entry(DocumentType.BOOK)]
        resolve_book_decisions(entries, settings("skip"))

        assert entries[0].decision.chapters is None

    def test_ask_with_no_way_to_ask_falls_back_to_the_cheap_default(self):
        """
        The case that makes this design safe.

        An unattended run, or a GUI that has not supplied its dialog, must
        never hang waiting for an answer nor quietly read a whole volume.
        """
        entries = [entry(DocumentType.BOOK)]
        resolve_book_decisions(entries, settings("ask"), prompt=None)

        assert entries[0].decision.chapters is None
        assert "no prompt available" in entries[0].decision.reason

    def test_ask_with_a_handler_asks(self):
        entries = [entry(DocumentType.BOOK)]
        resolve_book_decisions(entries, settings("ask"),
                               prompt=lambda e: BookDecision(chapters=[7], reason="asked"))

        assert entries[0].decision.chapters == [7]

    def test_a_gui_dialog_is_just_another_handler(self):
        """The seam that keeps the two front ends complementary."""
        asked = []

        def dialog(entry):
            asked.append(entry.path.name)
            return BookDecision(chapters=[2], reason="chosen in a dialog")

        entries = [entry(DocumentType.BOOK)]
        resolve_book_decisions(entries, settings("ask"), prompt=dialog)

        assert asked == ["book.pdf"]
        assert entries[0].decision.chapters == [2]

    def test_the_all_policy_never_prompts(self):
        entries = [entry(DocumentType.BOOK)]
        resolve_book_decisions(entries, settings("all"),
                               prompt=lambda e: pytest.fail("asked despite policy 'all'"))

        assert entries[0].decision.chapters == []

    def test_the_chapters_policy_uses_the_configured_numbers(self):
        entries = [entry(DocumentType.BOOK)]
        resolve_book_decisions(entries, settings("chapters", chapters=[3, 4]))

        assert entries[0].decision.chapters == [3, 4]

    def test_a_page_ceiling_applies_whatever_the_policy(self):
        entries = [entry(DocumentType.BOOK)]
        resolve_book_decisions(entries, settings("all", max_pages=40))

        assert entries[0].decision.max_pages == 40

    def test_a_ceiling_also_applies_to_a_prompted_answer(self):
        entries = [entry(DocumentType.BOOK)]
        resolve_book_decisions(entries, settings("ask", max_pages=40),
                               prompt=lambda e: BookDecision(chapters=[7], reason="asked"))

        assert entries[0].decision.max_pages == 40

    def test_articles_are_never_asked_about(self):
        entries = [entry(DocumentType.RESEARCH_ARTICLE, page_count=8, name="paper.pdf")]
        resolve_book_decisions(entries, settings("ask"),
                               prompt=lambda e: pytest.fail("asked about an article"))

        assert entries[0].decision is None

    def test_an_unreadable_pdf_is_not_asked_about(self):
        broken = entry(DocumentType.BOOK)
        broken.error = "PdfiumError"
        resolve_book_decisions([broken], settings("ask"),
                               prompt=lambda e: pytest.fail("asked about a broken PDF"))

        assert broken.decision is None

    def test_a_plausible_chapter_is_never_asked_about(self):
        """
        A chapter IS one chapter — there is nothing to ask.

        Every answer the question offers is phrased about a volume that
        CONTAINS chapters, and `router.build_strategy` gives
        BookChapterStrategy neither `chapters` nor `max_body_pages`, so no
        answer could take effect anyway.

        This was a live defect, found by driving the GUI by hand: the dialog
        offered "front matter only (fast, no GPU)" for a 36-page chapter and
        then read all 36 pages on the GPU. The CLI shared it and never showed
        it, because corpus runs pass --noninteractive and so supply no prompt.
        """
        entries = [entry(DocumentType.BOOK_CHAPTER, page_count=36,
                         name="chapter.pdf")]
        resolve_book_decisions(
            entries, settings("ask"),
            prompt=lambda e: pytest.fail("asked about a plausible chapter"))

        assert entries[0].decision is None

    def test_a_chapter_is_not_asked_about_under_any_policy(self):
        """Not just `ask` — no policy should invent a decision for a chapter
        that the strategy will then ignore."""
        for policy in ("skip", "all", "ask"):
            entries = [entry(DocumentType.BOOK_CHAPTER, page_count=36,
                             name="chapter.pdf")]
            resolve_book_decisions(entries, settings(policy))
            assert entries[0].decision is None, f"policy {policy!r} set a decision"


class TestTheMisclassifiedBookGuard:
    """
    A `book_chapter` of hundreds of pages is a book that scored wrongly.

    This matters because the opt-in protection covers the BOOK path only: a
    document classified `book_chapter` is handed to the article pipeline and
    read in full, with no ceiling. Without this guard a misclassification
    costs hours.
    """

    def test_an_implausibly_long_chapter_is_refused(self):
        entries = [entry(DocumentType.BOOK_CHAPTER, page_count=345)]
        resolve_book_decisions(entries, settings("all"),
                               prompt=lambda e: pytest.fail("asked, instead of refusing"))

        assert entries[0].decision.chapters is None
        assert "implausible" in entries[0].decision.reason

    def test_a_normal_chapter_is_unaffected(self):
        assert not entry(DocumentType.BOOK_CHAPTER, page_count=30).looks_misclassified

    @pytest.mark.parametrize("pages,flagged", [
        (IMPLAUSIBLE_CHAPTER_PAGES, False),
        (IMPLAUSIBLE_CHAPTER_PAGES + 1, True),
    ])
    def test_the_threshold(self, pages, flagged):
        assert entry(DocumentType.BOOK_CHAPTER, page_count=pages).looks_misclassified is flagged

    def test_a_long_book_is_not_flagged_as_misclassified(self):
        """A 345-page BOOK is just a book; only a long CHAPTER is suspicious."""
        assert not entry(DocumentType.BOOK, page_count=345).looks_misclassified


class TestThePrompt:
    ENTRY = PreflightEntry(path=Path("vol.pdf"), document_type=DocumentType.BOOK,
                           confidence=1.0, page_count=345,
                           title="Progress in Diseases", chapter_count=15)

    def _ask(self, *answers):
        replies = iter(answers)
        return ask_about_book(self.ENTRY, input_fn=lambda _: next(replies),
                              output_fn=lambda *args: None)

    def test_pressing_enter_takes_the_safe_default(self):
        assert self._ask("").chapters is None

    def test_option_two_reads_everything(self):
        assert self._ask("2").chapters == []

    def test_option_three_asks_which(self):
        assert self._ask("3", "3-5").chapters == [3, 4, 5]

    def test_an_unrecognised_answer_takes_the_safe_default(self):
        """A misread answer must never start hours of GPU work."""
        assert self._ask("banana").chapters is None

    def test_an_empty_chapter_list_falls_back_to_safe(self):
        assert self._ask("3", "none of them").chapters is None

    def test_a_closed_input_takes_the_safe_default(self):
        def closed(_):
            raise EOFError

        decision = ask_about_book(self.ENTRY, input_fn=closed,
                                  output_fn=lambda *args: None)
        assert decision.chapters is None

    def test_the_cost_is_shown_before_the_choice_is_made(self):
        lines = []
        ask_about_book(self.ENTRY, input_fn=lambda _: "",
                       output_fn=lambda *args: lines.append(" ".join(map(str, args))))
        shown = "\n".join(lines)

        assert "345" in shown and "15" in shown
        assert "Progress in Diseases" in shown
        assert "minutes of GPU" in shown


class TestChapterSelectionParsing:
    @pytest.mark.parametrize("text,expected", [
        ("3", [3]),
        ("3 4 5", [3, 4, 5]),
        ("3,4,5", [3, 4, 5]),
        ("3-5", [3, 4, 5]),
        ("1, 3-5", [1, 3, 4, 5]),
        ("5 3 3", [3, 5]),          # sorted and deduplicated
        ("", []),
        ("all of them", []),
        ("3-", []),                 # incomplete range, not a guess
    ])
    def test_selections(self, text, expected):
        assert parse_chapter_selection(text) == expected


class TestTheSummary:
    def test_the_summary_records_what_happened_to_each_document(self, tmp_path):
        entries = [entry(DocumentType.BOOK, name="vol.pdf"),
                   entry(DocumentType.RESEARCH_ARTICLE, page_count=8, name="paper.pdf")]
        entries[0].decision = BookDecision(reason="default")
        results = {str(entries[0].path): {"elapsed": 1.4, "output": "vol.json"},
                   str(entries[1].path): {"error": "boom"}}

        path = write_batch_summary(entries, results, tmp_path, elapsed=93.2)

        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["run"]["documents"] == 2
        assert payload["failed"] == ["paper.pdf"]
        assert payload["documents"][0]["book_decision"]["reason"] == "default"
        assert payload["documents"][1]["error"] == "boom"

    def test_both_a_stamped_and_a_stable_summary_are_written(self, tmp_path):
        write_batch_summary([], {}, tmp_path, elapsed=1.0)
        names = sorted(p.name for p in tmp_path.glob("*.json"))

        assert "batch_summary.json" in names
        assert any(name.startswith("batch_summary_2") for name in names)


class TestPreflightDisplay:
    def test_problem_documents_are_marked(self):
        broken = entry(DocumentType.BOOK, name="broken.pdf")
        broken.error = "PdfiumError"
        lines = []
        show_preflight([broken], output_fn=lambda *a: lines.append(" ".join(map(str, a))))

        assert any(line.strip().startswith("!") for line in lines)
