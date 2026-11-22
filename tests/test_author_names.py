"""
Each author's surname kept apart from the given names (contract 1.4,
`metadata_final.author_names`).

For biology researchers: an author list is stored as strings ("Sietske K.
Rosendahl Huber"), and no rule can tell where a compound surname begins: the
last word ("Huber") is wrong here, and "Rosendahl" could as well be a middle
name. Crossref keeps the surname as its own field; these tests check that the
field is carried from the registry record all the way to the output, and
dropped wherever it no longer describes the list.
"""

from src.processing.citations.chain import respell_authors
from src.processing.citations.crossref import canonical_record
from src.processing.document_json import with_author_names
from src.processing.ledger.semantic_ledger import LedgerRecord
from src.processing.review import apply_record_to_metadata
from src.processing.strategies.research_article.metadata_enrichment import (
    prefer_registry_authors)

CROSSREF_ITEM = {
    "title": ["Mucosal antibody responses"],
    "DOI": "10.1234/x",
    "author": [
        {"given": "Sietske K.", "family": "Rosendahl Huber"},
        {"given": "Diego", "family": "García Martínez de Artola"},
        {"given": "G", "family": "FULLERTON"},
        {"name": "The Consortium"},
        {"given": "", "family": ""},
    ],
}


class TestTheCrossrefRecord:
    def test_each_author_keeps_the_surname_apart(self):
        record = canonical_record(CROSSREF_ITEM)
        assert record["author_names"][:2] == [
            {"given": "Sietske K.", "family": "Rosendahl Huber"},
            {"given": "Diego", "family": "García Martínez de Artola"}]

    def test_names_and_strings_stay_one_to_one(self):
        record = canonical_record(CROSSREF_ITEM)
        assert len(record["author_names"]) == len(record["authors"]) == 4
        for part, name in zip(record["author_names"], record["authors"]):
            assert name.endswith(part["family"])

    def test_capitals_are_brought_to_normal_case_in_both(self):
        record = canonical_record(CROSSREF_ITEM)
        assert record["authors"][2] == "G Fullerton"
        assert record["author_names"][2] == {"given": "G", "family": "Fullerton"}


class TestTheTitlePageSpelling:
    def test_the_page_gives_the_given_names_the_registry_the_surname(self):
        record = {"authors": ["S K Rosendahl Huber"],
                  "author_names": [{"given": "S K", "family": "Rosendahl Huber"}]}
        respelled = respell_authors(record, {"authors": ["Sietske K. Rosendahl Huber"]})
        assert respelled["authors"] == ["Sietske K. Rosendahl Huber"]
        assert respelled["author_names"] == [{"given": "Sietske K.",
                                              "family": "Rosendahl Huber"}]

    def test_a_record_without_names_gets_none_added(self):
        respelled = respell_authors({"authors": ["Ann Lee"]}, {"authors": ["Ann Lee"]})
        assert "author_names" not in respelled


class TestTheMergedRecordAtOnline:
    def test_the_registry_surnames_come_with_its_list(self):
        record = canonical_record(CROSSREF_ITEM)
        merged = prefer_registry_authors({"authors": ["S. Huber"]}, record, "crossref_doi")
        assert merged["author_names"] == record["author_names"]

    def test_an_openalex_record_has_no_surnames_to_give(self):
        merged = prefer_registry_authors({"authors": ["Ann Lee"]}, {"authors": ["Ann Lee"]},
                                         "openalex_title")
        assert merged["author_names"] == []

    def test_a_zotero_record_changes_nothing(self):
        merged = prefer_registry_authors({"authors": ["Ann Lee"]},
                                         canonical_record(CROSSREF_ITEM), "zotero_doi")
        assert "author_names" not in merged


class TestTheWrittenDocument:
    def test_matching_names_are_written(self):
        metadata = {"authors": ["Diego García Martínez de Artola"],
                    "author_names": [{"given": "Diego", "family": "García Martínez de Artola"}]}
        assert with_author_names(metadata)["author_names"] == metadata["author_names"]

    def test_a_list_of_another_length_is_not_written(self):
        metadata = {"authors": ["Ann Lee", "Bo Li"],
                    "author_names": [{"given": "Ann", "family": "Lee"}]}
        assert with_author_names(metadata)["author_names"] == []

    def test_a_surname_that_does_not_end_its_name_is_not_written(self):
        metadata = {"authors": ["Ann Lee"], "author_names": [{"given": "Ann", "family": "Li"}]}
        assert with_author_names(metadata)["author_names"] == []

    def test_no_names_at_all_is_an_empty_list(self):
        assert with_author_names({"authors": ["Ann Lee"]})["author_names"] == []


class TestAHandCorrection:
    def record(self, authors):
        return LedgerRecord(document_id="h", pdf_filename="a.pdf", authors=authors,
                            validation_confidence=1.0, validation_source="human",
                            last_updated="2026-11-22T00:00:00", source_hash="h")

    def test_a_corrected_list_drops_the_registry_surnames(self):
        metadata = {"authors": ["Ann Lee"], "author_names": [{"given": "Ann", "family": "Lee"}]}
        patched = apply_record_to_metadata(self.record(["Ann Lee", "Bo Li"]), metadata)
        assert patched["author_names"] == []

    def test_an_unchanged_list_keeps_them(self):
        names = [{"given": "Ann", "family": "Lee"}]
        patched = apply_record_to_metadata(self.record(["Ann Lee"]),
                                           {"authors": ["Ann Lee"], "author_names": names})
        assert patched["author_names"] == names
