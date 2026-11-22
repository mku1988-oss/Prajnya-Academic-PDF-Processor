"""Stage S2: reference lists from Crossref, OpenAlex and the printed list.

No network: services are fakes.
"""
from src.processing.citations.crossref import deposited_reference
from src.processing.citations.http import STATUS_OK, STATUS_UNREACHABLE
from src.processing.citations.openalex import cited_work, printed_surname
from src.processing.citations.reference_lists import MAX_SINGLE_LOOKUPS, ReferenceListFinder
from src.processing.citations.reference_merge import merge_reference_lists
from src.processing.utils.reference_list import ReferenceList, make_entry

CONTRACT_KEYS = {"position", "number", "first_author", "year", "title", "container",
                 "doi", "source", "printed_text"}


def dep(author, year, title=None, doi=None, **extra):
    base = {"first_author": author, "year": year, "title": title, "container": None,
            "doi": doi, "volume": None, "first_page": None, "deposited_text": None}
    base.update(extra)
    return base


def local_list(rows, style="numbered"):
    """rows: (text) strings; numbered from 1."""
    entries = [make_entry(i if style == "numbered" else None, 1, text)
               for i, text in enumerate(rows, start=1)]
    return ReferenceList(style=style, entries=entries)


TEXTS = [
    "Bella, J. (2016) Collagen structure: new tricks. Biochem J 473, 1001",
    "Doi, M. (2005) Cell Biol Int 29, 12 Fibril assembly",
    "Fan, P. (2011) Matrix Biol 30, 55 Hydra regeneration",
    "Lee, K. (1999) Nature 400, 10 Wound healing",
    "Smith, A. (2001) Science 291, 33 Peptides",
]
CROSSREF = [dep("Bella", 2016), dep("Doi", 2005), dep("Fan", 2011),
            dep("Lee", 1999), dep("Smith", 2001)]


class TestMappings:
    def test_structured_deposited_reference(self):
        out = deposited_reference({"author": " Bella ", "year": "2016", "article-title": "A &amp; B",
                                   "journal-title": "Biochem J", "DOI": "10.1/x", "volume": "473",
                                   "first-page": "1001"})
        assert out["first_author"] == "Bella" and out["year"] == 2016
        assert out["title"] == "A & B" and out["container"] == "Biochem J"
        assert out["doi"] == "10.1/x" and out["volume"] == "473" and out["first_page"] == "1001"
        assert out["deposited_text"] is None

    def test_unstructured_only_item_reads_author_and_year_from_text(self):
        out = deposited_reference({"unstructured": "Doi, M., Nishi, Y. (2005) Cell Biol Int 29, 12"})
        assert out["first_author"] == "Doi" and out["year"] == 2005
        assert out["title"] is None and out["deposited_text"].startswith("Doi")

    def test_bare_doi_item(self):
        out = deposited_reference({"DOI": "10.1/abc"})
        assert out["doi"] == "10.1/abc"
        assert out["first_author"] is None and out["year"] is None

    def test_cited_work_with_van_der_surname(self):
        out = cited_work({"doi": "https://doi.org/10.1/Z", "title": "T", "publication_year": 2010,
                          "authorships": [{"author": {"display_name": "Annette van der Flier"}}],
                          "primary_location": {"source": {"display_name": "Cell"}},
                          "biblio": {"volume": "5", "first_page": "9"}})
        assert out["first_author"] == "van der Flier" and out["year"] == 2010
        assert out["doi"] == "10.1/Z" and out["container"] == "Cell"
        assert out["volume"] == "5" and out["first_page"] == "9"
        assert printed_surname("Jordi Bella") == "Bella"


class TestMergeWithCrossref:
    def test_crossref_only(self):
        result = merge_reference_lists(CROSSREF, [], None)
        block = result["reference_list"]
        assert block == {"status": "found", "source": "crossref", "count": 5,
                         "local_count": None, "counts_agree": None}
        assert all(r["number"] is None and r["source"] == "crossref" for r in result["references"])

    def test_agreeing_numbered_local_list(self):
        result = merge_reference_lists(CROSSREF, [], local_list(TEXTS))
        block, refs = result["reference_list"], result["references"]
        assert block["style"] == "numbered" and block["counts_agree"] is True
        assert [r["number"] for r in refs] == [1, 2, 3, 4, 5]
        assert refs[1]["printed_text"].startswith("Doi")
        assert all(set(r) == CONTRACT_KEYS for r in refs)

    def test_extra_local_entry_is_added_in_place(self):
        texts = TEXTS[:2] + ["Zed, Q. (2000) Extra entry the publisher omitted"] + TEXTS[2:]
        result = merge_reference_lists(CROSSREF, [], local_list(texts))
        refs = result["references"]
        assert [r["position"] for r in refs] == list(range(1, 7))
        assert refs[2]["source"] == "local" and refs[2]["number"] == 3
        assert refs[2]["first_author"] == "Zed" and refs[2]["year"] == 2000
        assert result["reference_list"]["counts_agree"] is False
        assert result["report"]["local_only_entries"] == 1

    def test_unrelated_local_list_is_not_trusted(self):
        other = local_list([f"Other{i}, X. (19{50 + i}) Something" for i in range(5)])
        result = merge_reference_lists(CROSSREF, [], other)
        refs = result["references"]
        assert len(refs) == 5 and all(r["number"] is None and r["source"] == "crossref" for r in refs)
        assert result["reference_list"]["local_count"] == 5

    def test_missing_author_filled_from_paired_local_entry(self):
        crossref = [dep("Bella", 2016), dep(None, None, doi="10.1/q"), dep("Fan", 2011),
                    dep("Lee", 1999), dep("Smith", 2001)]
        result = merge_reference_lists(crossref, [], local_list(TEXTS))
        second = result["references"][1]
        assert second["first_author"] == "Doi" and second["year"] == 2005
        assert second["doi"] == "10.1/q" and second["number"] == 2


class TestLocalSpine:
    def test_attach_by_title_evidence(self):
        work = dep("Fan", 2011, title="Hydra regeneration", doi="10.1/fan", container="Matrix Biol")
        result = merge_reference_lists([], [work], local_list(TEXTS))
        refs = result["references"]
        assert refs[2]["doi"] == "10.1/fan" and refs[2]["source"] == "openalex"
        assert refs[0]["source"] == "local" and refs[0]["doi"] is None
        assert result["report"]["attached"] == 1
        assert result["reference_list"]["source"] == "local"

    def test_refused_when_title_not_on_entry(self):
        work = dep("Fan", 2011, title="Completely unrelated study of ribosomes", doi="10.1/no")
        result = merge_reference_lists([], [work], local_list(TEXTS))
        assert result["report"]["attached"] == 0
        assert all(r["doi"] is None for r in result["references"])

    def test_tie_attaches_none(self):
        one = dep("Fan", 2011, title="Hydra regeneration", doi="10.1/a")
        two = dep("Fan", 2011, title="Hydra regeneration", doi="10.1/b")
        result = merge_reference_lists([], [one, two], local_list(TEXTS))
        assert result["report"]["attached"] == 0

    def test_local_only(self):
        result = merge_reference_lists([], [], local_list(TEXTS, style="author_year"))
        assert result["report"]["case"] == "local_only"
        assert all(r["number"] is None for r in result["references"])
        assert result["reference_list"]["style"] == "author_year"


class TestNotAvailable:
    def test_openalex_alone_gives_no_list(self):
        result = merge_reference_lists([], [dep("Fan", 2011)], None)
        assert result["reference_list"]["status"] == "not_available"
        assert result["references"] == [] and result["report"]["case"] == "openalex_unordered"

    def test_nothing_at_all(self):
        assert merge_reference_lists([], [], None)["report"]["case"] == "none"


class TestInitialsAndYears:
    def test_initials_beside_deposited_surname_are_dropped(self):
        assert deposited_reference({"author": "PA Tsonis", "year": "2004"})["first_author"] == "Tsonis"
        assert deposited_reference({"author": "Tsonis PA", "year": "2004"})["first_author"] == "Tsonis"

    def test_lone_capitalised_word_and_particles_are_left_alone(self):
        assert deposited_reference({"author": "AMANO"})["first_author"] == "AMANO"
        assert deposited_reference({"author": "van der Flier"})["first_author"] == "van der Flier"

    def test_printed_year_wins_when_one_apart(self):
        online = [dep("Bella", 2015), dep("Doi", 2005)]
        result = merge_reference_lists(online, [], local_list(TEXTS[:2]))
        assert [r["year"] for r in result["references"]] == [2016, 2005]

    def test_registry_year_kept_when_two_apart_without_other_tie(self):
        online = [dep("Bella", 2014), dep("Doi", 2005)]
        result = merge_reference_lists(online, [], local_list(TEXTS[:2]))
        assert result["references"][0]["year"] == 2014


AMBROSONE_LOCAL = [
    "Alpha, A. (2010) Mapping the early wound response. Dev Biol 5, 1",
    "Ambrosone, C. (2014) Breast tumour markers in cohorts. Cancer Res 74, 20",
    "Ambrosone, C. (2012) Genetic variants and survival. J Clin Oncol 30, 7",
    "Gamma, C. (2009) Unrelated entry on membranes. Cell 100, 3",
    "Beta, B. (2012) Regeneration signals in Hydra. Development 139, 40",
]


class TestPairingRules:
    def test_one_year_apart_pairs_even_with_missing_neighbours(self):
        online = [dep("Alpha", 2010), dep("Ambrosone", 2013), dep("Ambrosone", 2011),
                  dep("Beta", 2012)]
        result = merge_reference_lists(online, [], local_list(AMBROSONE_LOCAL))
        refs = result["references"]
        ambrosone = [r for r in refs if r["first_author"] == "Ambrosone"]
        assert len(ambrosone) == 2
        assert [r["year"] for r in ambrosone] == [2014, 2012]
        assert all(r["printed_text"] is not None for r in ambrosone)
        assert len(refs) == 5

    def test_deposited_title_printed_in_entry_pairs_despite_two_years(self):
        online = [dep("Bella", 2014, title="Collagen structure: new tricks"),
                  dep("Doi", 2005), dep("Fan", 2011), dep("Lee", 1999), dep("Smith", 2001)]
        result = merge_reference_lists(online, [], local_list(TEXTS))
        refs = result["references"]
        assert len(refs) == 5 and result["report"]["paired"] == 5
        assert refs[0]["printed_text"].startswith("Bella")
        assert refs[0]["year"] == 2014 and refs[0]["source"] == "crossref"


class TestNumbersAgainstShorterList:
    def test_numbers_given_when_registry_holds_only_part_of_list(self):
        names = ["Aa", "Bb", "Cc", "Dd", "Ee", "Ff", "Gg", "Hh", "Ii", "Jj"]
        texts = [f"{name}, X. ({2000 + i}) Study number {i} Journal {i}, 1"
                 for i, name in enumerate(names)]
        online = [dep(name, 2000 + i) for i, name in enumerate(names) if i not in (2, 6)]
        result = merge_reference_lists(online, [], local_list(texts))
        refs = result["references"]
        assert len(refs) == 10 and result["report"]["paired"] == 8
        assert [r["number"] for r in refs] == list(range(1, 11))
        assert [r["source"] for r in refs if r["number"] in (3, 7)] == ["local", "local"]
        assert result["reference_list"]["counts_agree"] is False


class FakeCrossref:
    def __init__(self, status=STATUS_OK, entries=(), records=None):
        self.status, self.entries = status, list(entries)
        self.records, self.by_doi_calls = records, []

    def deposited_references(self, doi):
        return self.status, [dict(e) for e in self.entries]

    def by_doi(self, doi):
        self.by_doi_calls.append(doi)
        return STATUS_OK, self.records


class FakeOpenAlex:
    def __init__(self, status=STATUS_OK, found=None, cited=()):
        self.status, self.found, self.cited, self.asked = status, found or {}, list(cited), []

    def works_by_doi(self, dois):
        self.asked.append(list(dois))
        return self.status, self.found

    def cited_works(self, doi):
        return self.status, list(self.cited)


class TestFinder:
    def test_bare_dois_are_filled_from_openalex(self):
        crossref = FakeCrossref(entries=[dep(None, None, doi="10.1/Q"), dep("Fan", 2011, title="T")])
        openalex = FakeOpenAlex(found={"10.1/q": dep("Qin", 2012, title="Qin title", doi="10.1/Q")})
        finder = ReferenceListFinder(crossref, openalex, local_reader=lambda path: None)
        result = finder.find("10.1/paper", None)
        first = result["references"][0]
        assert (first["first_author"], first["year"], first["title"]) == ("Qin", 2012, "Qin title")
        assert openalex.asked == [["10.1/Q"]]

    def test_doi_only_entry_unknown_to_openalex_is_identified_by_crossref(self):
        record = {"authors": ["Hiroaki Imataka"], "year": 1992, "title": "A title", "journal": "EMBO J"}
        crossref = FakeCrossref(entries=[dep(None, None, doi="10.1/I"), dep("Fan", 2011, title="T")],
                                records=record)
        finder = ReferenceListFinder(crossref, FakeOpenAlex(), local_reader=lambda path: None)
        first = finder.find("10.1/paper", None)["references"][0]
        assert first["first_author"] == "Imataka" and first["year"] == 1992
        assert crossref.by_doi_calls == ["10.1/I"]

    def test_single_lookups_are_capped(self):
        unknown = [dep(None, None, doi=f"10.1/u{i}") for i in range(MAX_SINGLE_LOOKUPS + 5)]
        crossref = FakeCrossref(entries=unknown, records=None)
        finder = ReferenceListFinder(crossref, FakeOpenAlex(), local_reader=lambda path: None)
        finder.find("10.1/paper", None)
        assert len(crossref.by_doi_calls) == MAX_SINGLE_LOOKUPS

    def test_unreachable_services_give_not_reached(self):
        finder = ReferenceListFinder(FakeCrossref(status=STATUS_UNREACHABLE),
                                     FakeOpenAlex(status=STATUS_UNREACHABLE),
                                     local_reader=lambda path: None)
        assert finder.find("10.1/paper", None)["reference_list"]["status"] == "not_reached"

    def test_no_doi_uses_local_list_only(self):
        crossref = FakeCrossref(entries=CROSSREF)
        finder = ReferenceListFinder(crossref, FakeOpenAlex(), local_reader=lambda path: local_list(TEXTS))
        result = finder.find(None, None)
        assert result["report"]["case"] == "local_only"
        assert result["reference_list"]["source"] == "local" and len(result["references"]) == 5
