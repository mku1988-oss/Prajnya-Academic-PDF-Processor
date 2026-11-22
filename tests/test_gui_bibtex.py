#!/usr/bin/env python3
"""Tests for gui/bibtex.py — the BibTeX export extracted from V5's
ledger_master.py and made document-type-aware for S8."""

from src.gui import bibtex
from src.processing.ledger.semantic_ledger import LedgerRecord


def _record(**overrides) -> LedgerRecord:
    defaults = dict(
        document_id="d1", pdf_filename="paper.pdf",
        validation_confidence=0.9, validation_source="test",
        last_updated="2026-08-11T00:00:00", source_hash="d1",
    )
    defaults.update(overrides)
    return LedgerRecord(**defaults)


class TestGenerateCiteKey:
    def test_lastname_journal_year(self):
        record = _record(authors=["Smith, John"], journal="Nature", year=2023)
        assert bibtex.generate_cite_key(record) == "smith_nature_2023"

    def test_no_comma_takes_first_word(self):
        record = _record(authors=["Smith"], journal="Science", year=2020)
        assert bibtex.generate_cite_key(record) == "smith_science_2020"

    def test_missing_year_is_nd(self):
        record = _record(authors=["Doe, Jane"], journal="Science")
        assert bibtex.generate_cite_key(record) == "doe_science_nd"

    def test_missing_author_and_journal(self):
        record = _record()
        assert bibtex.generate_cite_key(record) == "anonymous_unknown_nd"

    def test_book_falls_back_to_editors(self):
        record = _record(authors=[], editors=["Roberts, A."], publisher="Springer", year=2019)
        assert bibtex.generate_cite_key(record) == "roberts_springer_2019"

    def test_accents_are_transliterated(self):
        record = _record(authors=["Müller, K."], journal="Über", year=2021)
        key = bibtex.generate_cite_key(record)
        assert key == "muller_uber_2021"


class TestBuildBibtex:
    def test_duplicate_cite_keys_get_disambiguated(self):
        records = [
            _record(document_id="a", authors=["Smith, J."], journal="Nature", year=2023),
            _record(document_id="b", authors=["Smith, J."], journal="Nature", year=2023),
        ]
        content = bibtex.build_bibtex(records)
        assert "@article{smith_nature_2023," in content
        assert "@article{smith_nature_2023_2," in content

    def test_article_uses_author_field(self):
        record = _record(authors=["Doe, Jane"], journal="Cell", year=2020,
                         document_type="research_article")
        content = bibtex.build_bibtex([record])
        assert content.startswith("@article{")
        assert "author={Doe, Jane}," in content

    def test_book_uses_book_type_and_editor_field(self):
        record = _record(authors=[], editors=["Roberts, A.", "Halper, J."],
                         publisher="Springer", year=2019, document_type="book")
        content = bibtex.build_bibtex([record])
        assert content.startswith("@book{")
        assert "editor={Roberts, A. and Halper, J.}," in content
        assert "publisher={Springer}," in content

    def test_book_chapter_uses_incollection(self):
        record = _record(authors=["Wheeler, T."], journal=None, publisher="Elsevier",
                         year=2018, document_type="book_chapter")
        content = bibtex.build_bibtex([record])
        assert content.startswith("@incollection{")

    def test_missing_doi_is_notfound(self):
        record = _record(authors=["Doe, Jane"], journal="Cell", year=2020, doi=None)
        content = bibtex.build_bibtex([record])
        assert "doi={notfound}," in content

    def test_special_characters_are_escaped(self):
        record = _record(authors=["A & B"], title="50% faster with $x$ & {y}",
                         journal="J", year=2020)
        content = bibtex.build_bibtex([record])
        assert r"50\% faster with \$x\$ \& \{y\}" in content
        assert r"A \& B" in content

    def test_empty_list_produces_empty_string(self):
        assert bibtex.build_bibtex([]) == ""
