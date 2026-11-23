"""
Tests for ledger T34 (v6.9 item 3): the journal name at level `local`.

For biology researchers: a running header read as the journal loses its
volume and date tail; a section word read as the journal ("REVIEWS") is kept
but flags the record for review; an abbreviation and a journal printed in
capitals are left alone.
"""
import logging

from src.processing.citations.journal_name import check_journal_name
from src.processing.review import assess_metadata_completeness
from src.processing.strategies.research_article.metadata_enrichment import (annotate_provenance,
                                                                            apply_citation_level)


class TestCheckJournalName:
    def test_a_running_header_tail_is_cut(self):
        assert check_journal_name("TRENDS in Genetics Vol.20 No.1 January 2004") == ("TRENDS in Genetics", False)

    def test_a_volume_issue_year_tail_is_cut(self):
        assert check_journal_name("Journal of Cell Science, Volume 12, Issue 3, 2019") == ("Journal of Cell Science", False)

    def test_a_section_word_is_flagged(self):
        assert check_journal_name("REVIEWS") == ("REVIEWS", True)
        assert check_journal_name("Research Article") == ("Research Article", True)

    def test_abbreviations_and_capitalised_journals_are_left_alone(self):
        for name in ("Biochem. J.", "GENETICS", "CELL", "Physiological Reviews", "Nature Reviews Genetics"):
            assert check_journal_name(name) == (name, False)

    def test_nothing_is_cut_to_empty(self):
        assert check_journal_name("Vol. 3") == ("Vol. 3", False)


class TestAtLevelLocal:
    def record(self, journal):
        return {"title": "Hydration-induced changes", "authors": ["Quan H"], "doi": "10.1/x",
                "journal": journal, "doi_source": "pymupdf_text_layer", "doi_confidence": 0.98}

    def test_a_section_word_flags_the_record(self):
        merged = apply_citation_level(self.record("REVIEWS"), "local", "Hydration-induced changes Quan",
                                      "q.pdf", logger=logging.getLogger("t"))
        merged["provenance"] = annotate_provenance(merged)
        assessment = assess_metadata_completeness(merged)
        assert assessment.requires_review and "journal" in assessment.suspicious_items

    def test_a_registry_record_is_not_touched(self):
        merged = apply_citation_level(self.record("REVIEWS"), "zotero", "", "q.pdf",
                                      logger=logging.getLogger("t"))
        assert merged["journal"] == "REVIEWS" and "suspicious_journal" not in merged
