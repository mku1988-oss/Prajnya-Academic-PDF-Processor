"""Citation level `online` (stage S1): acceptance rule, cache, chain, merge.

No test touches the network: services are fakes, and the HTTP client is fed
by a stubbed `requests.get`.
"""
import json

from src.processing.citations import http as http_module
from src.processing.citations.acceptance import (check_candidate, choose_record,
                                                  clean_author_list,
                                                  complete_truncated_authors,
                                                  is_truncated_author_list, surname_of,
                                                  title_similarity)
from src.processing.citations.chain import OnlineChain, respell_authors
from src.processing.citations.crossref import canonical_record as crossref_record
from src.processing.citations.http import (STATUS_NOT_FOUND, STATUS_OK,
                                           STATUS_UNREACHABLE, CitationHttp)
from src.processing.citations.openalex import canonical_record as openalex_record
from src.processing.strategies.research_article.metadata_enrichment import establish_metadata

TITLE = "Collagen structure: new tricks from a very old dog"
READ = {"title": TITLE, "authors": ["Jordi Bella"], "year": 2016, "doi": "10.1042/BJ20160953"}
WRONG_TITLE = "A different paper about kinases entirely"


def rec(title=TITLE, authors=("Jordi Bella",), year=2016, doi="10.1042/bj20160953",
        type_="journal-article"):
    return {"title": title, "authors": list(authors), "year": year, "doi": doi,
            "type": type_, "journal": "Biochemical Journal", "publisher": "Portland Press"}


class FakeService:
    def __init__(self, name, by_doi=None, by_title=None, status=STATUS_OK):
        self.name, self._doi, self._title, self.status = name, by_doi, by_title or [], status
        self.calls = []

    def by_doi(self, doi):
        self.calls.append(("doi", doi))
        return self.status, self._doi

    def by_title(self, title, surname=""):
        self.calls.append(("title", title))
        return self.status, self._title


class TestAcceptanceRule:
    def test_a_matching_record_is_accepted(self):
        assert check_candidate(rec(), READ)[0]

    def test_title_below_090_is_refused(self):
        assert not check_candidate(rec(title="Collagen: an unrelated review"), READ)[0]

    def test_wrong_first_author_is_refused(self):
        assert not check_candidate(rec(authors=["Someone Else"]), READ)[0]

    def test_the_2014_edition_is_refused_on_its_year(self):
        assert not check_candidate(rec(year=2014), READ)[0]

    def test_print_and_online_years_one_apart_pass(self):
        assert check_candidate(rec(year=2017), READ)[0]

    def test_no_author_read_then_year_must_match(self):
        reading = {**READ, "authors": []}
        assert check_candidate(rec(), reading)[0]
        assert not check_candidate(rec(year=2017), reading)[0]

    def test_neither_author_nor_year_read_is_rejected(self):
        assert not check_candidate(rec(), {"title": TITLE})[0]

    def test_component_record_is_skipped_when_a_paper_exists(self):
        supporting = rec(type_="component", doi="10.1042/bj20160953.s1")
        chosen, _ = choose_record([supporting, rec()], READ)
        assert chosen["doi"] == "10.1042/bj20160953"

    def test_two_different_dois_that_both_pass_are_ambiguous(self):
        chosen, why = choose_record([rec(doi="10.1/a"), rec(doi="10.1/b")], READ)
        assert chosen is None and "ambiguous" in why

    def test_a_doi_is_checked_against_the_title_page_text(self):
        garbled = {**READ, "title": "garbled"}
        ok, _ = check_candidate(rec(title=WRONG_TITLE), garbled,
                                title_page_text="x", by_doi=True)
        assert not ok
        ok, _ = check_candidate(rec(), garbled, title_page_text=TITLE, by_doi=True)
        assert ok

    def test_surnames_and_similarity_fold_case_accents_and_markup(self):
        assert surname_of("FULLERTON, G") == "fullerton"
        assert surname_of("Annette van der Flier") == "van der flier"
        assert title_similarity("Café <i>E. coli</i>", "cafe e coli") == 1.0

    def test_affiliation_marks_glued_to_a_name_are_ignored(self):
        """Seen on the Bella paper: the model read "Jordi Bella*1"."""
        assert surname_of("Jordi Bella*1") == "bella"
        assert surname_of("Lee¹, K.") == "lee"
        assert check_candidate(rec(), {**READ, "authors": ["Jordi Bella*1"]})[0]

    def test_a_whole_author_line_read_as_one_string_still_matches(self):
        """Seen twice in the S1 confirmation: one string holding every author."""
        line = "Roland Aufschnaiter1,*‡, Evan A. Zamir2, Charles D. Little1"
        assert check_candidate(rec(authors=["Roland Aufschnaiter", "Evan Zamir"]),
                               {**READ, "authors": [line]})[0]
        line2 = "Xiaoming Zhang†*, Raymond P. Boot-Handford‡1,2"
        assert check_candidate(rec(authors=["Xiaoming Zhang"]), {**READ, "authors": [line2]})[0]
        assert not check_candidate(rec(authors=["Someone Else"]), {**READ, "authors": [line]})[0]

    def test_last_comma_first_still_reads_as_a_surname(self):
        assert check_candidate(rec(), {**READ, "authors": ["Bella, Jordi"]})[0]

    def test_review_articles_are_not_discarded_as_non_papers(self):
        assert choose_record([rec(type_="review")], READ)[0] is not None


class TestCleanAuthorList:
    def test_marks_and_leading_and_are_removed_and_joined_names_split(self):
        raw = ["Xiaoming Zhang†*, Raymond P. Boot-Handford‡1,2", "Julie Huxley-Jones§3",
               "and Michael P. Sarras, Jr.*9"]
        assert clean_author_list(raw) == ["Xiaoming Zhang", "Raymond P. Boot-Handford",
                                          "Julie Huxley-Jones", "Michael P. Sarras, Jr."]

    def test_numbered_byline_in_one_string_is_split(self):
        # Benchmark pilot PMC10904388 (Nat Commun): "1,2" leaves ",," behind,
        # and the last two names are joined by "and".
        raw = ["Paula Istvan1,2*, Einar Birkeland1,2†, Willem M. de Vos8,9, "
               "Paula Berstad4 and Trine B. Roughe2,3,11"]
        assert clean_author_list(raw) == ["Paula Istvan", "Einar Birkeland",
                                          "Willem M. de Vos", "Paula Berstad",
                                          "Trine B. Roughe"]

    def test_superscript_and_plus_marks_are_removed(self):
        # Pilots PMC8937232 (eLife) and PMC8942261 (PLOS Pathogens).
        raw = ["Yixiao Zhang¹²*, Lisha Sun¹,2,3†, Caigang Liu¹,2,3*",
               "Marion Lussignol¹†‡, Audrey Esclatine¹**+"]
        assert clean_author_list(raw) == ["Yixiao Zhang", "Lisha Sun", "Caigang Liu",
                                          "Marion Lussignol", "Audrey Esclatine"]

    def test_unsplittable_and_is_left_alone(self):
        assert clean_author_list(["Smith, J and Jones, K"]) == ["Smith, J and Jones, K"]

    def test_ellipsis_of_a_shortened_list_is_not_a_name(self):
        raw = ["Margaret E. Kirkling", "Urszula Cytlak", "...", "Boris Reizis"]
        assert clean_author_list(raw) == ["Margaret E. Kirkling", "Urszula Cytlak",
                                          "Boris Reizis"]
        assert clean_author_list(["A Bee, C Dee, ..., E Eff"]) == ["A Bee", "C Dee", "E Eff"]

    def test_last_comma_first_is_not_split(self):
        assert clean_author_list(["Bella, Jordi"]) == ["Bella, Jordi"]

    def test_non_lists_give_an_empty_list(self):
        assert clean_author_list(None) == [] and clean_author_list("x") == []


class TestRecordMapping:
    def test_crossref_initials_in_capitals_are_tidied_and_respelled(self):
        item = {"title": [TITLE], "DOI": "10.1/x", "type": "journal-article",
                "author": [{"given": "G", "family": "FULLERTON"}],
                "container-title": ["J"], "issued": {"date-parts": [[2006]]},
                "published-print": {"date-parts": [[2007]]}}
        record = crossref_record(item)
        assert record["authors"] == ["G Fullerton"] and record["year"] == 2007
        spelled = respell_authors(record, {"authors": ["Gary D. Fullerton"]})
        assert spelled["authors"] == ["Gary D. Fullerton"]

    def test_respelling_never_copies_affiliation_marks(self):
        record = {"authors": ["G Fullerton"]}
        spelled = respell_authors(record, {"authors": ["Gary D. Fullerton*1"]})
        assert spelled["authors"] == ["Gary D. Fullerton"]

    def test_a_lower_cased_doi_keeps_the_printed_case(self):
        spelled = respell_authors(rec(doi="10.1042/bj20151169"),
                                  {"authors": [], "doi": "10.1042/BJ20151169"})
        assert spelled["doi"] == "10.1042/BJ20151169"
        other = respell_authors(rec(doi="10.1/zzz"), {"authors": [], "doi": "10.1042/BJ20151169"})
        assert other["doi"] == "10.1/zzz"

    def test_html_entities_in_registry_text_are_unescaped(self):
        item = {"title": ["Cells &amp; Development"], "container-title": ["Cells &amp; Dev"],
                "publisher": "A &amp; B", "author": [], "DOI": "10.1/x"}
        record = crossref_record(item)
        assert record["title"] == "Cells & Development" and record["publisher"] == "A & B"

    def test_openalex_record_has_a_bare_doi(self):
        work = {"display_name": TITLE, "doi": "https://doi.org/10.1/x", "publication_year": 2016,
                "authorships": [{"author": {"display_name": "Jordi Bella"}}],
                "primary_location": {"source": {"display_name": "Biochem J"}}, "type": "article"}
        record = openalex_record(work)
        assert record["doi"] == "10.1/x" and record["journal"] == "Biochem J"


class TestChain:
    def test_doi_accepted_at_crossref_never_asks_openalex(self):
        crossref, openalex = FakeService("crossref", rec()), FakeService("openalex")
        out = OnlineChain([crossref, openalex]).get_canonical_metadata(READ, TITLE)
        assert out["source"] == "crossref_doi" and out["confidence"] == 0.95
        assert openalex.calls == []

    def test_wrong_doi_is_dropped_and_the_title_path_is_tried(self):
        crossref = FakeService("crossref", rec(title=WRONG_TITLE), by_title=[rec()])
        out = OnlineChain([crossref]).get_canonical_metadata(READ, TITLE)
        assert out["source"] == "crossref_title" and out["record"]["doi"]

    def test_wrong_doi_with_no_title_hit_asks_for_the_doi_to_be_dropped(self):
        crossref = FakeService("crossref", rec(title=WRONG_TITLE))
        out = OnlineChain([crossref]).get_canonical_metadata(READ, TITLE)
        assert out["record"] == {} and out["drop_doi"] and out["network"] == "ok"

    def test_openalex_is_the_second_opinion(self):
        crossref = FakeService("crossref", None, status=STATUS_NOT_FOUND)
        openalex = FakeService("openalex", rec())
        out = OnlineChain([crossref, openalex]).get_canonical_metadata(READ, TITLE)
        assert out["source"] == "openalex_doi"

    def test_dead_network_is_not_the_same_as_no_record(self):
        dead = FakeService("crossref", None, status=STATUS_UNREACHABLE)
        out = OnlineChain([dead]).get_canonical_metadata(READ, TITLE)
        assert out["network"] == "unavailable" and not out["drop_doi"]
        absent = FakeService("crossref", None, status=STATUS_NOT_FOUND)
        assert OnlineChain([absent]).get_canonical_metadata(READ, TITLE)["network"] == "ok"


class TestEstablishMetadataOnline:
    PAGE = f"{TITLE}\nJordi Bella\nBiochemical Journal 2016"

    def _run(self, service):
        return establish_metadata(dict(READ), self.PAGE, citation_level="online",
                                  file_name="b.pdf", online_chain=OnlineChain([service]))

    def test_accepted_record_sets_source_and_confidence(self):
        merged = self._run(FakeService("crossref", rec()))
        assert merged["validation_source"] == "crossref_doi"
        assert merged["validation_confidence"] == 0.95
        assert merged["journal"] == "Biochemical Journal"

    def test_dead_network_stands_on_own_reading_and_is_flagged(self):
        merged = self._run(FakeService("crossref", None, status=STATUS_UNREACHABLE))
        assert merged["validation_confidence"] <= 0.60
        assert merged["provenance"]["network"] == "unavailable"
        assert "network" not in merged

    def test_a_wrong_doi_does_not_reach_the_final_record(self):
        merged = self._run(FakeService("crossref", rec(title=WRONG_TITLE)))
        assert merged["doi"] is None

    def test_the_registry_author_list_replaces_a_misread_byline(self):
        """The byline was read 'Stefan Hümer' with a glued mark and a name
        missed; the accepted Crossref record lists all three, in order, and
        that list is the one kept (benchmark: 29 -> 43 of 50 exact)."""
        read = {**READ, "authors": ["Jordi Bella", "Stefan Hümer"]}
        page = f"{TITLE}\nJordi Bella, Ana Ruiz, Stefan Hümmer\nBiochemical Journal 2016"
        record = rec(authors=("Jordi Bella", "Ana Ruiz", "Stefan Hümmer"))
        merged = establish_metadata(read, page, citation_level="online", file_name="b.pdf",
                                    online_chain=OnlineChain([FakeService("crossref", record)]))
        assert merged["authors"] == ["Jordi Bella", "Ana Ruiz", "Stefan Hümmer"]

    def test_a_registry_list_cut_off_after_the_first_author_is_not_used(self):
        """Crossref's record for 10.1016/j.tig.2003.11.004 names only
        "J Myllyharju"; the byline names two. The cut-off list must not drop
        Kivirikko (v6.8 release check)."""
        read = {**READ, "authors": ["Jordi Bella", "Ana Ruiz", "Stefan Hümmer"]}
        page = f"{TITLE}\nJordi Bella, Ana Ruiz, Stefan Hümmer\nBiochemical Journal 2016"
        record = rec(authors=("J Bella",))
        merged = establish_metadata(read, page, citation_level="online", file_name="b.pdf",
                                    online_chain=OnlineChain([FakeService("crossref", record)]))
        assert [a.split()[-1] for a in merged["authors"]] == ["Bella", "Ruiz", "Hümmer"]
        assert merged["validation_source"] == "crossref_doi"     # the record still vouches

    def test_with_no_record_accepted_the_reading_stands(self):
        read = {**READ, "authors": ["Jordi Bella", "Stefan Hümer"]}
        merged = establish_metadata(read, self.PAGE, citation_level="online", file_name="b.pdf",
                                    online_chain=OnlineChain([FakeService("crossref", None)]))
        assert [a.split()[-1] for a in merged["authors"]] == ["Bella", "Hümer"]


class TestHttpCache:
    def _client(self, tmp_path):
        return CitationHttp(tmp_path, pacing_seconds=0)

    def _stub(self, monkeypatch, status, text="{}"):
        calls = []

        class Response:
            headers = {}

        Response.status_code, Response.text = status, text

        def fake_get(url, **kwargs):
            calls.append(url)
            return Response()
        monkeypatch.setattr(http_module.requests, "get", fake_get)
        return calls

    def test_a_200_is_cached_and_asked_once(self, tmp_path, monkeypatch):
        calls = self._stub(monkeypatch, 200, json.dumps({"a": 1}))
        client = self._client(tmp_path)
        assert client.get_json("https://x/y") == (STATUS_OK, {"a": 1})
        assert client.get_json("https://x/y") == (STATUS_OK, {"a": 1})
        assert len(calls) == 1

    def test_a_404_is_cached_too(self, tmp_path, monkeypatch):
        calls = self._stub(monkeypatch, 404)
        client = self._client(tmp_path)
        client.get_json("https://x/y")
        assert client.get_json("https://x/y")[0] == STATUS_NOT_FOUND and len(calls) == 1

    def test_a_5xx_is_never_cached(self, tmp_path, monkeypatch):
        calls = self._stub(monkeypatch, 503)
        client = self._client(tmp_path)
        assert client.get_json("https://x/y")[0] == STATUS_UNREACHABLE
        client.get_json("https://x/y")
        assert len(calls) == 2

    def test_a_timeout_is_never_cached(self, tmp_path, monkeypatch):
        def boom(url, **kwargs):
            raise http_module.requests.exceptions.ConnectTimeout()
        monkeypatch.setattr(http_module.requests, "get", boom)
        client = self._client(tmp_path)
        assert client.get_json("https://x/y")[0] == STATUS_UNREACHABLE
        assert list(tmp_path.glob("*.json")) == []

    def test_an_api_key_is_sent_but_never_written_to_disk(self, tmp_path, monkeypatch):
        calls = self._stub(monkeypatch, 200, "{}")
        client = self._client(tmp_path)
        client.get_json("https://x/y", {"api_key": "SECRET", "q": "t"})
        assert "SECRET" in calls[0]
        assert all("SECRET" not in f.read_text(encoding="utf-8") for f in tmp_path.glob("*.json"))
        client.get_json("https://x/y", {"api_key": "OTHER", "q": "t"})
        assert len(calls) == 1      # a changed key does not discard good answers


# Benchmark pilot PMC9396529 (Cell Reports): the summary page shortens the
# author list, the next page prints the byline in full (text layer as read).
CELL_SUMMARY_AUTHORS = ["Biana Bernshtein, Esther Ndungo, Deniz Cizmeci, ..., "
                        "Karen L. Kotloff, Marcela F. Pasetti, Galit Alter"]
CELL_NEXT_PAGE = ("Article\nSystems approach to define humoral correlates\n"
                  "Biana Bernshtein,1,7 Esther Ndungo,2,7 Deniz Cizmeci,1 Peng Xu,3 "
                  "Pavol Kova´ c,  3 Meagan Kelly,4 Dilara Islam,2\nEdward T. Ryan,4,5 "
                  "Karen L. Kotloff,2 Marcela F. Pasetti,2,6 and Galit Alter1,6,8,* "
                  "1Ragon Institute of MGH, Harvard and MIT, Cambridge, MA, USA")


class TestShortenedAuthorLists:
    def test_a_list_with_an_ellipsis_is_truncated(self):
        assert is_truncated_author_list(CELL_SUMMARY_AUTHORS)
        assert is_truncated_author_list(["A Bee", "…", "C Dee"])
        assert not is_truncated_author_list(["A Bee", "C Dee"])
        assert not is_truncated_author_list(None)

    def test_full_byline_is_read_from_the_next_page(self):
        full = complete_truncated_authors(CELL_SUMMARY_AUTHORS, CELL_NEXT_PAGE)
        assert full == ["Biana Bernshtein", "Esther Ndungo", "Deniz Cizmeci", "Peng Xu",
                        "Pavol Kovác", "Meagan Kelly", "Dilara Islam",
                        "Edward T. Ryan", "Karen L. Kotloff", "Marcela F. Pasetti",
                        "Galit Alter"]

    def test_no_completion_when_the_byline_is_not_on_the_next_page(self):
        assert complete_truncated_authors(CELL_SUMMARY_AUTHORS, "Results and discussion") is None
        assert complete_truncated_authors(CELL_SUMMARY_AUTHORS, "") is None

    def test_no_completion_when_a_shown_name_is_missing_from_the_byline(self):
        other = CELL_NEXT_PAGE.replace("Esther Ndungo", "Someone Else")
        assert complete_truncated_authors(CELL_SUMMARY_AUTHORS, other) is None

    def test_truncated_list_is_completed_at_local(self):
        reading = {"title": "Systems approach to define humoral correlates of immunity to Shigella",
                   "authors": CELL_SUMMARY_AUTHORS}
        merged = establish_metadata(reading, title_page_text="Systems approach",
                                    citation_level="local", next_page_text=CELL_NEXT_PAGE)
        assert len(merged["authors"]) == 11
        assert not merged["provenance"].get("authors_truncated")

    def test_uncompleted_truncated_list_is_flagged(self):
        reading = {"title": "Systems approach to define humoral correlates of immunity to Shigella",
                   "authors": CELL_SUMMARY_AUTHORS}
        merged = establish_metadata(reading, title_page_text="Systems approach",
                                    citation_level="local")
        assert merged["provenance"]["authors_truncated"] is True
        assert "..." not in merged["authors"]
