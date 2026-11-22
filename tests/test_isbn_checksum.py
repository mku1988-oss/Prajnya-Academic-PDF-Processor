"""An unlabelled number is an ISBN only if its check digit is right.

Session 33: a Nature Communications article (Wnt3/Sp5, Hydra) was classified
as a BOOK. Springer Nature PDFs carry a hidden "1234567890():,;" string on
every page; the bare ISBN-10 pattern took those ten digits as an ISBN, and the
book path then wrote a record with no title.
"""
from src.processing.extractors.metadata_extractor import MetadataExtractor, is_valid_isbn

NATURE_FOOTER = ("NATURE COMMUNICATIONS | (2019) 10:312 | "
                 "https://doi.org/10.1038/s41467-018-08242-2 | "
                 "www.nature.com/naturecommunications 1\r\n1234567890():,;")


class TestCheckDigit:
    def test_real_isbns_pass(self):
        assert is_valid_isbn("0262033844")        # ISBN-10
        assert is_valid_isbn("9780262033848")     # ISBN-13
        assert is_valid_isbn("080442957X")        # ISBN-10 with X check digit

    def test_runs_of_digits_that_are_not_isbns_fail(self):
        assert not is_valid_isbn("1234567890")    # the Springer Nature font string
        assert not is_valid_isbn("9780262033849")
        assert not is_valid_isbn("12345")
        assert not is_valid_isbn("X234567890")


class TestBareIsbnNeedsAValidCheckDigit:
    def setup_method(self):
        self.extractor = MetadataExtractor()

    def test_the_nature_footer_has_no_isbn(self):
        assert self.extractor.extract_isbn_from_text(NATURE_FOOTER) == (None, 0.0)

    def test_a_real_isbn_after_a_stray_number_is_still_found(self):
        text = "page code 1234567890 ... printed as 0-262-03384-4"
        assert self.extractor.extract_isbn_from_text(text) == ("0262033844", 0.85)

    def test_a_scientific_reports_volume_marker_is_not_an_isbn(self):
        """'0123456789' passes the check digit; it is a Springer Nature marker."""
        text = "1\r\nVol.:(0123456789)\r\nScientific Reports"
        assert self.extractor.extract_isbn_from_text(text) == (None, 0.0)

    def test_a_page_range_in_a_reference_is_not_an_isbn(self):
        """'28523-28526' joins to 2852328526, which passes the check digit by chance."""
        text = "J Biol Chem\r\n270: 28523-28526."
        assert self.extractor.extract_isbn_from_text(text) == (None, 0.0)

    def test_a_pnas_doi_suffix_is_not_an_isbn(self):
        assert self.extractor.extract_isbn_from_text("doi 10.1073/pnas.2220167120") == (None, 0.0)

    def test_a_bare_isbn13_is_found(self):
        assert self.extractor.extract_isbn_from_text("978-0-262-03384-8") == ("9780262033848", 0.85)

    def test_a_labelled_isbn_is_unchanged(self):
        assert self.extractor.extract_isbn_from_text("ISBN: 978-0-262-03384-8") == ("9780262033848", 0.95)
