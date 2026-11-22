"""Stage S3a: numbered in-text markers in the `citations` block.

No PDF and no network. `search_numeric_markers` (the linker) is replaced by a
stand-in where the test is about what this stage does with its answer.
"""
from types import SimpleNamespace

import pytest

from src.processing.citations import markers as markers_module
from src.processing.citations.block import build_citations_block, reference_data_of
from src.processing.citations.markers import locate_markers, numeric_markers_for
from src.processing.citations.reference_lists import ReferenceListFinder
from src.processing.utils.citation_markers import CitationMarker, MarkerSearch


def marker(text, context, kind="bracket", page=1, numbers=(1,)):
    return {"page_number": page, "text": text, "kind": kind, "numbers": list(numbers),
            "context_before": context, "content_offset": None}


def offset_of(one_marker, content, page=1):
    return locate_markers([one_marker], {page: content})[0]["content_offset"]


# --- locate_markers ---------------------------------------------------------------------

class TestLocateMarkers:
    def test_found_once(self):
        content = "Collagen is a protein [3] found in skin."
        assert offset_of(marker("[3]", "Collagen is a protein "), content) == content.index("[3]")

    def test_found_twice_with_different_contexts_picks_the_right_one(self):
        content = "Skin is strong [4]. Bone is hard [4]."
        second = marker("[4]", "Bone is hard ")
        assert offset_of(second, content) == content.rindex("[4]")

    def test_found_twice_with_the_same_context_is_none(self):
        content = "as shown [4]. as shown [4]."
        assert offset_of(marker("[4]", "as shown "), content) is None

    def test_two_identical_markers_both_get_none(self):
        content = "as shown [4]. as shown [4]."
        both = locate_markers([marker("[4]", "as shown ")] * 2, {1: content})
        assert [m["content_offset"] for m in both] == [None, None]

    def test_superscript_is_not_found_inside_longer_numbers(self):
        mark = marker("245", "was reported", kind="superscript", numbers=(245,))
        assert offset_of(mark, "was reported1245 and later") is None
        assert offset_of(mark, "was reported2456 and later") is None
        assert offset_of(mark, "was reported245 and later") == len("was reported")

    def test_any_dash_matches_any_dash(self):
        content = "in vitro [3–7] and in vivo."
        assert offset_of(marker("[3-7]", "in vitro "), content) == content.index("[")

    def test_hyphenated_line_break_in_the_context_is_harmless(self):
        content = "the fibril assem- bly [5] is slow"
        assert offset_of(marker("[5]", "the fibril assembly "), content) == content.index("[5]")

    def test_markdown_marks_and_spaces_in_the_context_are_harmless(self):
        content = "**Collagen**   is  a protein [3]"
        assert offset_of(marker("[3]", "Collagen is a protein "), content) == content.index("[3]")

    def test_context_that_disagrees_gives_none(self):
        assert offset_of(marker("[3]", "something else entirely "), "Collagen is a protein [3]") is None

    def test_missing_page_gives_none(self):
        assert locate_markers([marker("[3]", "x ", page=9)], {1: "x [3]"})[0]["content_offset"] is None

    def test_empty_context_with_a_single_occurrence(self):
        assert offset_of(marker("[3]", ""), "text [3] more") == 5

    def test_empty_context_with_two_occurrences_is_none(self):
        assert offset_of(marker("[3]", ""), "text [3] more [3]") is None

    def test_the_input_is_not_changed(self):
        original = marker("[3]", "")
        locate_markers([original], {1: "[3]"})
        assert original["content_offset"] is None

    def test_a_malformed_marker_never_raises(self):
        assert locate_markers([{"page_number": 1}], {1: "x"})[0]["content_offset"] is None


# --- numeric_markers_for ----------------------------------------------------------------

def reference(number, position=None):
    return {"position": position or number, "number": number}


def merged_list(numbers=(1, 2, 3), status="found", counts_agree=True, positions=None):
    refs = [reference(n, (positions or {}).get(n)) for n in numbers]
    block = {"status": status, "style": "numbered", "count": len(refs)}
    if counts_agree is not None:
        block["counts_agree"] = counts_agree
    return {"reference_list": block, "references": refs}


NUMBERED = SimpleNamespace(style="numbered", entries=[], first_line=0)
LINES = ["a line"]


@pytest.fixture
def fake_search(monkeypatch):
    """Replace the linker; the stand-in records how it was called."""
    calls = []

    def install(found, dropped=None):
        def search(lines, reference_list, last=None, list_confirmed=False):
            calls.append({"last": last, "list_confirmed": list_confirmed})
            return MarkerSearch(list(found), dict(dropped or {"beyond_list": 0, "order_check": 0}))
        monkeypatch.setattr(markers_module, "search_numeric_markers", search)
        return calls
    return install


def found_marker(numbers, text="[1]", kind="bracket"):
    return CitationMarker(page=2, text=text, kind=kind, numbers=list(numbers), context="before ")


class TestNumericMarkersFor:
    def test_author_year_list_gives_none(self, fake_search):
        fake_search([found_marker([1])])
        author_year = SimpleNamespace(style="author_year", entries=[], first_line=0)
        found, report = numeric_markers_for(LINES, author_year, merged_list())
        assert found == [] and report["reason"] == "author-year list"

    def test_no_printed_numbers_gives_none(self, fake_search):
        fake_search([found_marker([1])])
        merged = merged_list()
        for ref in merged["references"]:
            ref["number"] = None
        found, report = numeric_markers_for(LINES, NUMBERED, merged)
        assert found == [] and report["reason"] == "no printed numbers"

    def test_no_lines_no_local_list_or_no_found_list_gives_none(self, fake_search):
        fake_search([found_marker([1])])
        assert numeric_markers_for([], NUMBERED, merged_list())[0] == []
        assert numeric_markers_for(LINES, None, merged_list())[0] == []
        assert numeric_markers_for(LINES, NUMBERED, merged_list(status="not_available"))[0] == []

    def test_a_number_without_a_linkable_reference_is_dropped_and_counted(self, fake_search):
        fake_search([found_marker([1]), found_marker([2, 9], text="[2,9]")])
        found, report = numeric_markers_for(LINES, NUMBERED, merged_list())
        assert [m["numbers"] for m in found] == [[1]]
        assert report["dropped"]["not_in_list"] == 1 and report["found"] == 1

    def test_a_number_that_is_not_its_position_is_not_linkable(self, fake_search):
        calls = fake_search([found_marker([2], text="[2]")])
        found, report = numeric_markers_for(LINES, NUMBERED, merged_list(positions={2: 3}))
        assert found == [] and report["dropped"]["not_in_list"] == 1
        assert calls[0]["last"] == 3          # largest linkable number: 1, 3

    @pytest.mark.parametrize("agree, expected", [(True, True), (False, False), (None, False)])
    def test_counts_agree_is_passed_as_list_confirmed(self, fake_search, agree, expected):
        calls = fake_search([found_marker([1])])
        _, report = numeric_markers_for(LINES, NUMBERED, merged_list(counts_agree=agree))
        assert calls[0]["list_confirmed"] is expected and report["list_confirmed"] is expected

    def test_a_marker_has_the_contract_shape_and_reading_order(self, fake_search):
        fake_search([found_marker([1]), found_marker([3], text="[3]")],
                    dropped={"beyond_list": 4, "order_check": 2})
        found, report = numeric_markers_for(LINES, NUMBERED, merged_list())
        assert found[0] == {"page_number": 2, "text": "[1]", "kind": "bracket", "numbers": [1],
                            "context_before": "before ", "content_offset": None}
        assert [m["text"] for m in found] == ["[1]", "[3]"]
        assert report["style"] == "bracket"
        assert report["dropped"] == {"beyond_list": 4, "order_check": 2, "author_line": 0, "not_in_list": 0}


# --- the finder -------------------------------------------------------------------------

class QuietCrossref:
    def deposited_references(self, doi):
        return "ok", []

    def by_doi(self, doi):
        return "ok", None


class QuietOpenAlex:
    def works_by_doi(self, dois):
        return "ok", {}

    def cited_works(self, doi):
        return "ok", []


def finder_with(**kwargs):
    return ReferenceListFinder(QuietCrossref(), QuietOpenAlex(), **kwargs)


class TestFinderMarkers:
    def test_markers_key_is_there_when_there_is_no_pdf(self):
        result = finder_with(line_reader=lambda path: None).find(None, None)
        assert result["markers"] == []
        assert result["report"]["markers"]["found"] == 0

    def test_markers_key_is_there_with_a_doi_and_no_list(self):
        result = finder_with(line_reader=lambda path: None).find("10.1/x", None)
        assert result["markers"] == [] and "markers" in result["report"]

    def test_lines_are_read_once_and_used_for_the_markers(self, monkeypatch):
        reads = []
        listing = SimpleNamespace(style="numbered", entries=[], first_line=0)
        monkeypatch.setattr(markers_module, "search_numeric_markers",
                            lambda *a, **k: MarkerSearch([], {"beyond_list": 0, "order_check": 0}))
        monkeypatch.setattr("src.processing.citations.reference_lists.list_from_lines",
                            lambda lines: listing if lines else None)
        finder = finder_with(line_reader=lambda path: reads.append(path) or ["a line"])
        finder.find(None, "paper.pdf")
        assert reads == ["paper.pdf"]

    def test_an_error_in_the_marker_step_leaves_the_list_intact(self, monkeypatch):
        def broken(lines, local_list, merged):
            raise RuntimeError("boom")
        monkeypatch.setattr("src.processing.citations.reference_lists.numeric_markers_for", broken)
        result = finder_with(line_reader=lambda path: None).find(None, None)
        assert result["markers"] == []
        assert result["reference_list"]["status"] == "not_available"
        assert result["report"]["markers"]["reason"] == "error while linking"

    def test_the_style_of_the_local_list_picks_the_linker(self, monkeypatch):
        calls = []
        stub = {"found": 0, "style": None, "dropped": {}, "list_confirmed": False, "reason": "stub"}
        monkeypatch.setattr("src.processing.citations.reference_lists.numeric_markers_for",
                            lambda *a: calls.append("numeric") or ([], stub))
        monkeypatch.setattr("src.processing.citations.reference_lists.author_year_markers_for",
                            lambda *a: calls.append("author_year") or ([], stub))
        for style in ("numbered", "author_year"):
            listing = SimpleNamespace(style=style, entries=[], first_line=0)
            monkeypatch.setattr("src.processing.citations.reference_lists.list_from_lines",
                                lambda lines, listing=listing: listing)
            finder_with(line_reader=lambda path: ["a line"]).find(None, "paper.pdf")
        assert calls == ["numeric", "author_year"]

    def test_an_author_year_paper_gets_markers_through_the_finder(self, monkeypatch):
        body = [SimpleNamespace(page=1, text="as shown (Doi et al., 2005)."),
                SimpleNamespace(page=2, text="Doi M, Nishi Y, Lee K. 2005. A list.")]
        listing = SimpleNamespace(style="author_year", entries=[], first_line=1)
        monkeypatch.setattr("src.processing.citations.reference_lists.list_from_lines", lambda lines: listing)
        monkeypatch.setattr("src.processing.citations.reference_lists.merge_reference_lists",
                            lambda *a: {"reference_list": {"status": "found", "counts_agree": True, "count": 1, "local_count": 1,
                                                            "source": "crossref"},
                                        "references": [{"position": 1, "number": None, "first_author": "Doi",
                                                        "year": 2005, "printed_text": "Doi M, Nishi Y, Lee K. 2005."}],
                                        "report": {"case": "x"}})
        result = finder_with(line_reader=lambda path: body).find(None, "paper.pdf")
        assert [(m["kind"], m["numbers"]) for m in result["markers"]] == [("author_year", [1])]
        assert result["report"]["markers"]["found"] == 1

    def test_local_reader_still_supplies_the_list(self):
        called = []
        finder = finder_with(local_reader=lambda path: called.append(path), line_reader=lambda p: None)
        finder.find(None, "paper.pdf")
        assert called == ["paper.pdf"]


# --- the block --------------------------------------------------------------------------

DATA = {"reference_list": {"status": "found", "source": "crossref", "count": 1},
        "references": [{"position": 1, "number": 1}],
        "markers": [marker("[1]", "Collagen is a protein ")]}
PAGES = {1: "Collagen is a protein [1] found in skin."}


class TestBlockMarkers:
    def test_markers_carry_offsets_when_page_contents_are_given(self):
        block = build_citations_block({}, "x.pdf", "online", "cli", reference_data=DATA,
                                      page_contents=PAGES)
        assert block["markers"][0]["content_offset"] == PAGES[1].index("[1]")

    def test_without_page_contents_the_offsets_are_none(self):
        block = build_citations_block({}, "x.pdf", "online", "cli", reference_data=DATA)
        assert len(block["markers"]) == 1 and block["markers"][0]["content_offset"] is None

    def test_markers_are_empty_without_reference_data(self):
        assert build_citations_block({}, "x.pdf", page_contents=PAGES)["markers"] == []

    def test_the_finders_markers_are_not_changed(self):
        build_citations_block({}, "x.pdf", "online", "cli", reference_data=DATA, page_contents=PAGES)
        assert DATA["markers"][0]["content_offset"] is None

    def test_reference_data_of_carries_markers_forward_and_offsets_follow_the_new_text(self):
        block = build_citations_block({}, "x.pdf", "online", "cli", reference_data=DATA,
                                      page_contents=PAGES)
        carried = reference_data_of(block)
        assert carried["markers"] == block["markers"]
        moved = {1: "New first sentence. Collagen is a protein [1] found in skin."}
        again = build_citations_block({}, "x.pdf", "online", "cli", reference_data=carried,
                                      page_contents=moved)
        assert again["markers"][0]["content_offset"] == moved[1].index("[1]")
