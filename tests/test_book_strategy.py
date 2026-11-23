#!/usr/bin/env python3
"""
Tests for the book path: TOC parsing, the front-matter record, and routing.

The gate for this path is `Tendon assembly 2021 [1-7].pdf` — the front matter
of a Springer volume — with its known-good output established by hand from the
printed pages, the same way the NMR baseline was. The expectations below are
that known-good record:

    title      Progress in Heritable Soft Connective Tissue Diseases
    editors    Jaroslava Halper
    publisher  Springer Nature Switzerland AG
    year       2021
    series     Advances in Experimental Medicine and Biology
    ISBN       978-3-030-80613-2   (copyright page, p5)
    DOI        10.1007/978-3-030-80614-9
    contents   15 chapters, starting at pages 1, 5, 45, 105, 127 …

For contrast, V5's `BookStrategy` returned `title = "Tendon assembly 2021 [1-7]"`
and no identifiers at all.

The parsing tests use text fixtures taken verbatim from that PDF's text layer,
including the details that broke the first implementation: a chapter title
wrapped over four lines, an entry whose leaders are only three dots, and a
continuation line ("Laminin, Elastin, Fibrillins, Fibulins, Matrilins,") that
looks exactly like an author list.
"""

from pathlib import Path

import pytest

from src.processing.models import DocumentProfile, DocumentType
from src.processing.strategies.book import (
    BookStrategy,
    build_book_record,
    parse_toc_entries,
)
from src.processing.strategies.router import build_strategy
from tests.local_samples import SAMPLES as LOCAL_SAMPLES

SAMPLES = LOCAL_SAMPLES
SPRINGER_BOOK = SAMPLES / "Extracted" / "Tendon assembly 2021 [1-7].pdf"
CONFIG_ROOT = Path(__file__).parent.parent / "config"

needs_springer = pytest.mark.skipif(not SPRINGER_BOOK.is_file(),
                                    reason="Springer sample PDF not available")

# Verbatim from page 6 of the sample's text layer.
REAL_TOC = """v
1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
Jaroslava Halper
2 Basic Structure, Physiology, and Biochemistry
of Connective Tissues and Extracellular Matrix Collagens . . . 5
Michael J. Mienaltowski, Nicole L. Gonzales,
Jessica M. Beall, and Monica Y. Pechanec
3 Tendon Extracellular Matrix Assembly,
Maintenance and Dysregulation Throughout Life . . . . . . . . . 45
Seyed Mohammad Siadat, Danae E. Zamboulis,
Chavaunne T. Thorpe, Jeffrey W. Ruberti,
and Brianne K. Connizzo
4 Basic Components of Connective Tissues
and Extracellular Matrix: Fibronectin, Fibrinogen,
Laminin, Elastin, Fibrillins, Fibulins, Matrilins,
Tenascins and Thrombospondins . . . . . . . . . . . . . . . . . 105
Jaroslava Halper
"""

REAL_TITLE_PAGE = """Jaroslava Halper
Editor
Progress in Heritable
Soft Connective
Tissue Diseases
Second Edition
"""

REAL_COPYRIGHT_PAGE = """ISSN 0065-2598 ISSN 2214-8019 (electronic)
Advances in Experimental Medicine and Biology
ISBN 978-3-030-80613-2 ISBN 978-3-030-80614-9 (eBook)
https://doi.org/10.1007/978-3-030-80614-9
© Springer Nature Switzerland AG 2021
This work is subject to copyright. All rights are reserved by the Publisher.
"""


class TestTocParsing:
    def test_every_chapter_is_found_in_printed_order(self):
        entries = parse_toc_entries(REAL_TOC)
        assert [entry.number for entry in entries] == [1, 2, 3, 4]
        assert [entry.start_page for entry in entries] == [1, 5, 45, 105]

    def test_a_title_wrapped_over_two_lines_is_rejoined(self):
        entry = parse_toc_entries(REAL_TOC)[1]
        assert entry.title == ("Basic Structure, Physiology, and Biochemistry "
                               "of Connective Tissues and Extracellular Matrix Collagens")

    def test_a_title_wrapped_over_four_lines_is_rejoined(self):
        """The line 'Laminin, Elastin, …' looks like an author list but is not."""
        entry = parse_toc_entries(REAL_TOC)[3]
        assert entry.title.startswith("Basic Components of Connective Tissues")
        assert entry.title.endswith("Tenascins and Thrombospondins")

    def test_the_previous_entrys_authors_are_not_swallowed(self):
        for entry in parse_toc_entries(REAL_TOC):
            assert "Halper" not in entry.title
            assert "Mienaltowski" not in entry.title

    def test_three_spaced_dots_are_enough_to_be_leaders(self):
        """Chapter 2's leaders are only '. . . 5' — measured, not hypothetical."""
        entries = parse_toc_entries("of Connective Tissues and Collagens . . . 5\n")
        assert len(entries) == 1
        assert entries[0].start_page == 5

    def test_an_ellipsis_in_prose_is_not_an_entry(self):
        assert parse_toc_entries("As shown ... in 2018 the fibrils align.\n") == []

    @pytest.mark.parametrize("line,number,title,page", [
        ("Index . . . . . . . . . . 337\n", None, "Index", 337),
        ("Preface..........ix\n", None, "Preface", None),
        ("12 Meester-Loeys Syndrome . . . . . 265\n", 12, "Meester-Loeys Syndrome", 265),
    ])
    def test_entry_shapes(self, line, number, title, page):
        entry = parse_toc_entries(line)[0]
        assert (entry.number, entry.title, entry.start_page) == (number, title, page)

    def test_a_page_with_no_leaders_yields_nothing(self):
        assert parse_toc_entries("Chapter 1\nIntroduction\nSome prose.\n") == []


class TestChapterAuthorship:
    """
    Each chapter of an edited volume has DIFFERENT authors.

    A retrieval system quoting chapter 3 must cite Siadat et al., not the
    book's editor Halper. The contents page states that mapping outright and
    this is the only cheap place to read it — once the body pages are being
    processed, the front matter is out of scope.
    """

    def test_each_chapter_carries_its_own_authors(self):
        entries = parse_toc_entries(REAL_TOC)
        assert entries[0].authors == ["Jaroslava Halper"]
        assert entries[1].authors == ["Michael J. Mienaltowski", "Nicole L. Gonzales",
                                      "Jessica M. Beall", "Monica Y. Pechanec"]

    def test_the_oxford_comma_does_not_glue_and_to_the_last_name(self):
        entry = parse_toc_entries(REAL_TOC)[2]
        assert entry.authors[-1] == "Brianne K. Connizzo"
        assert not any(name.startswith("and ") for name in entry.authors)

    def test_authors_are_not_confused_with_the_next_entrys_title(self):
        """
        Chapter 15's authors are followed directly by an unnumbered entry:

            Jennifer Hope Roberts and Jaroslava Halper
            Correction to: Clinical and Molecular Delineation of Cutis
            Laxa Syndromes: … . . . . . C1
        """
        page = ("15 Connective Tissue Disorders in Domestic Animals . . . . 325\n"
                "Jennifer Hope Roberts and Jaroslava Halper\n"
                "Correction to: Clinical and Molecular Delineation of Cutis\n"
                "Laxa Syndromes: Paradigms for Elastic Fiber Homeostasis . . . C1\n")
        entries = parse_toc_entries(page)
        assert entries[0].authors == ["Jennifer Hope Roberts", "Jaroslava Halper"]
        assert entries[1].authors == []

    def test_an_index_entry_has_no_authors(self):
        assert parse_toc_entries("Index . . . . . . 337\n")[0].authors == []

    @pytest.mark.parametrize("line,expected", [
        ("Lut Van Laer, Bart Loeys", ["Lut Van Laer", "Bart Loeys"]),
        ("Pauline De Kinderen", ["Pauline De Kinderen"]),
        ("Jean-Luc de la Tour", ["Jean-Luc de la Tour"]),
    ])
    def test_nobiliary_particles_stay_inside_the_surname(self, line, expected):
        page = f"1 A Chapter . . . . . 1\n{line}\n2 Next . . . . . 9\n"
        assert parse_toc_entries(page)[0].authors == expected

    @pytest.mark.parametrize("not_a_name_list", [
        "Correction to: Clinical Delineation",   # a colon
        "Volume 130 of the series",              # a digit
        "and the following chapters cover",      # lowercase words
    ])
    def test_lines_that_are_not_names_end_the_author_list(self, not_a_name_list):
        page = f"1 A Chapter . . . . . 1\n{not_a_name_list}\n"
        assert parse_toc_entries(page)[0].authors == []

    def test_affiliation_markers_are_stripped(self):
        """The same cleaning the article path applies, so keys match Zotero."""
        page = "1 A Chapter . . . . . 1\nJordi Bella*1, Karl E. Kadler2\n2 B . . . 9\n"
        assert parse_toc_entries(page)[0].authors == ["Jordi Bella", "Karl E. Kadler"]

    def test_a_malformed_page_cannot_swallow_the_column_as_authors(self):
        """
        The line budget is the backstop.

        On a well-formed page the name-list guard stops collection long before
        this matters. It exists for a page where it does not — a column of
        name-like lines with no next entry to end them.
        """
        from src.processing.strategies.book.toc import MAX_AUTHOR_LINES

        # One name per line, so the LINE budget is visible in the name count.
        page = "1 A Chapter . . . . . 1\n" + "".join(
            f"Given{chr(65 + i)} Surname{chr(65 + i)},\n" for i in range(10))
        authors = parse_toc_entries(page)[0].authors
        assert len(authors) == MAX_AUTHOR_LINES
        assert authors[0] == "GivenA SurnameA"

    def test_author_order_is_preserved(self):
        """Order is data in biology — it encodes contribution."""
        entry = parse_toc_entries(REAL_TOC)[2]
        assert entry.authors[0] == "Seyed Mohammad Siadat"
        assert entry.authors[-1] == "Brianne K. Connizzo"


class TestBookRecord:
    def test_the_record_is_read_off_the_front_matter(self):
        record = build_book_record({4: REAL_TITLE_PAGE, 5: REAL_COPYRIGHT_PAGE},
                                   copyright_page=5)
        assert record.title == "Progress in Heritable Soft Connective Tissue Diseases"
        assert record.editors == ["Jaroslava Halper"]
        assert record.publisher == "Springer Nature Switzerland AG"
        assert record.year == 2021
        assert record.series == "Advances in Experimental Medicine and Biology"
        assert (record.title_page, record.copyright_page) == (4, 5)

    def test_the_copyright_page_is_found_by_its_isbn_when_not_given(self):
        record = build_book_record({4: REAL_TITLE_PAGE, 5: REAL_COPYRIGHT_PAGE})
        assert record.copyright_page == 5
        assert record.title_page == 4

    def test_the_editor_role_and_edition_are_not_part_of_the_title(self):
        record = build_book_record({4: REAL_TITLE_PAGE, 5: REAL_COPYRIGHT_PAGE},
                                   copyright_page=5)
        assert "Editor" not in record.title
        assert "Edition" not in record.title
        assert "Halper" not in record.title

    def test_the_inline_editor_layout_is_handled(self):
        """Springer's half-title prints 'Jaroslava Halper   Editor' on one line."""
        record = build_book_record({1: "Jaroslava Halper   Editor\nProgress in Diseases\n"})
        assert record.editors == ["Jaroslava Halper"]

    def test_several_editors_are_split(self):
        record = build_book_record({1: "Ann Lee, Bo Chen and Cara Diaz\nEditors\nA Title\n"})
        assert record.editors == ["Ann Lee", "Bo Chen", "Cara Diaz"]

    def test_the_series_is_not_confused_with_an_adjacent_isbn_line(self):
        """Some copyright pages print ISSN, then ISBN, then the series name."""
        record = build_book_record({1: (
            "ISSN 0065-2598 (print)\n"
            "ISBN 978-3-030-80613-2\n"
            "Advances in Experimental Medicine and Biology\n")})
        assert record.series == "Advances in Experimental Medicine and Biology"

    def test_a_book_that_states_nothing_gets_none_not_a_guess(self):
        record = build_book_record({1: "   \n"})
        assert (record.title, record.publisher, record.year) == (None, None, None)


class TestRouting:
    def test_a_book_routes_to_the_book_strategy(self):
        strategy = build_strategy(DocumentType.BOOK, CONFIG_ROOT)
        assert type(strategy).__name__ == "BookStrategy"

    def test_an_unclassified_document_routes_to_the_article_strategy(self, monkeypatch):
        """None is the classifier's safe default and must not crash routing."""
        import src.processing.strategies.router as router

        built = {}

        class FakeArticleStrategy:
            def __init__(self, *args, **kwargs):
                built["called"] = True

        monkeypatch.setattr(
            "src.processing.strategies.research_article.strategy.ResearchArticleStrategy",
            FakeArticleStrategy)
        strategy = router.build_strategy(None, CONFIG_ROOT)
        assert isinstance(strategy, FakeArticleStrategy)

    def test_the_book_path_does_not_build_the_article_pipeline(self, monkeypatch):
        """A book run must not pay for the config tree and the ONNX model."""
        def explode(*args, **kwargs):
            raise AssertionError("the article strategy was constructed for a book")

        monkeypatch.setattr(
            "src.processing.strategies.research_article.strategy.ResearchArticleStrategy",
            explode)
        assert type(build_strategy(DocumentType.BOOK, CONFIG_ROOT)).__name__ == "BookStrategy"


@needs_springer
class TestTheBookGate:
    """The known-good record for the Springer volume, established by hand."""

    @pytest.fixture()
    def result(self):
        from run_prajnya import build_document_profile
        from src.processing.pdf import get_backend

        backend = get_backend()
        profile = build_document_profile(SPRINGER_BOOK, backend)
        strategy = BookStrategy(pdf_backend=backend)
        return strategy.process(profile).processing_result

    def test_the_bibliographic_record(self, result):
        metadata = result.metadata
        assert metadata.title == "Progress in Heritable Soft Connective Tissue Diseases"
        assert metadata.editors == ["Jaroslava Halper"]
        assert metadata.publisher == "Springer Nature Switzerland AG"
        assert metadata.year == 2021
        assert metadata.isbn == "9783030806132"
        assert metadata.doi == "10.1007/978-3-030-80614-9"

    def test_the_series_and_provenance(self, result):
        provenance = result.metadata.provenance
        assert provenance["series"] == "Advances in Experimental Medicine and Biology"
        assert provenance["issn"] == "0065-2598"
        assert provenance["copyright_page"] == 5
        assert "chapter bodies not extracted" in provenance["note"]

    def test_the_contents_map(self, result):
        assert result.toc_pages == [6, 7]
        # 15 chapters, plus the correction notice and the index.
        assert len(result.toc) == 17
        assert result.toc[0] == "1. Introduction"
        assert result.toc[14].startswith("15. Connective Tissue Disorders in Domestic Animals")

    def test_the_chapter_records_carry_per_chapter_authorship(self, result):
        """The mapping a RAG index needs to cite a passage to the right people."""
        chapters = {record["number"]: record for record in result.chapters
                    if record["number"] is not None}
        assert len(chapters) == 15
        assert all(record["authors"] for record in chapters.values())

        assert chapters[3]["authors"][0] == "Seyed Mohammad Siadat"
        assert chapters[3]["start_page_printed"] == 45
        assert chapters[10]["authors"] == ["Noriko Miyake", "Tomoki Kosho",
                                           "Naomichi Matsumoto"]
        # The editor wrote some chapters but not others — which is exactly why
        # book-level metadata cannot stand in for chapter attribution.
        assert chapters[1]["authors"] == ["Jaroslava Halper"]
        assert "Jaroslava Halper" not in chapters[10]["authors"]

    def test_chapter_start_pages_are_printed_numbers_not_pdf_indices(self, result):
        """
        The book's own page numbering, which is NOT the PDF page index.

        Chapter 3 starts on printed page 45, which is PDF page 53 in the full
        volume — the front matter offsets them. Resolving that offset needs the
        body pages and belongs to the book reading pipeline.
        """
        chapters = {record["number"]: record for record in result.chapters
                    if record["number"] is not None}
        assert [chapters[n]["start_page_printed"] for n in (1, 2, 3, 4)] == [1, 5, 45, 105]

    def test_the_front_matter_pages_are_returned_as_content(self, result):
        assert [page.page_number for page in result.pages] == [1, 2, 3, 4, 5, 6, 7]
        roles = {page.page_number: page.metadata["role"] for page in result.pages}
        assert roles[5] == "front_matter"
        assert roles[6] == "contents"

    def test_the_run_reports_no_errors(self, result):
        assert result.errors == []

    def test_the_title_is_not_the_filename(self, result):
        """V5's BookStrategy set title = pdf_path.stem. This is the fix."""
        assert result.metadata.title != SPRINGER_BOOK.stem


@needs_springer
class TestEndToEndRouting:
    def test_the_entrypoint_routes_the_springer_volume_to_the_book_strategy(self):
        from run_prajnya import build_document_profile
        from src.processing.pdf import get_backend

        profile = build_document_profile(SPRINGER_BOOK, get_backend())
        strategy = build_strategy(profile.document_type, CONFIG_ROOT)
        assert type(strategy).__name__ == "BookStrategy"
