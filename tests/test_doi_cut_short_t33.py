"""
Tests for ledger T33 (v6.9 item 3): a DOI cut short at a line end.

For biology researchers: PNAS prints its DOI at the end of a line in the
supplement link ("doi:10.1073/pnas." then "2220167120" on the next line),
and whole in its footer. The DOI read must be the whole one, and nothing
may be invented by joining text across lines.
"""
from src.processing.extractors.metadata_extractor import MetadataExtractor

PNAS_PAGE = ("information online\r\nat http://www.pnas.org/lookup/suppl/doi:10.1073/pnas.\r\n"
             "2220167120/-/DCSupplemental.\r\nPublished March 22, 2023.\r\n"
             "PNAS 2023 Vol. 120 No. 13 e2220167120 https://doi.org/10.1073/pnas.2220167120 1 of 8")


def doi_of(text):
    return MetadataExtractor().extract_doi_from_text(text)


class TestDoiCutShort:
    def test_the_whole_doi_printed_elsewhere_on_the_page_is_taken(self):
        assert doi_of(PNAS_PAGE) == ("10.1073/pnas.2220167120", 0.98)

    def test_a_cut_doi_with_no_whole_copy_is_left_as_found(self):
        assert doi_of("lookup/suppl/doi:10.1073/pnas.\n2220167120/-/DCSupplemental.")[0] == "10.1073/pnas"

    def test_a_different_doi_that_merely_shares_a_prefix_is_not_taken(self):
        # "10.1038/nature1" and "10.1038/nature12345": the continuation is a digit, not a separator.
        text = "doi: 10.1038/nature1 and a citation 10.1038/nature12345"
        assert doi_of(text)[0] == "10.1038/nature1"

    def test_a_whole_doi_is_unchanged(self):
        assert doi_of("doi: 10.1016/j.cellbi.2005.09.009") == ("10.1016/j.cellbi.2005.09.009", 0.98)
