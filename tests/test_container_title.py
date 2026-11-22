#!/usr/bin/env python3
"""
A book chapter's container is not called "journal", and we were only asking
for "journal".

For biology researchers: the *container* is the larger work a document sits
inside. For a research article that is the journal; for a chapter it is the
volume — "Collagen Fibril Assembly and Function" appears **in** "Current
Topics in Developmental Biology". A citation without it is incomplete.

THE DEFECT. Zotero stores the container under a different key per item type,
and `_canonicalize` read only the article one:

    journalArticle  publicationTitle    <- the only key we read
    bookSection     bookTitle           <- the chapter's container, dropped
    conferencePaper proceedingsTitle

So the Elsevier chapter's container title arrived in **every** API response
and was thrown away. Worse, the loss was self-concealing: the completeness
rule for a chapter accepts `journal|publisher`, Zotero also supplies
`publisher: Elsevier`, and the either-or group was satisfied by the other
branch. The record read "metadata complete" while a field we already had in
hand was missing — the flag was gone but the gap was not.

The payloads below are the real cached Zotero responses for two documents in
the sample corpus, trimmed to the fields under test.
"""

import pytest

from src.processing.utils.zotero_service import (
    CONTAINER_TITLE_KEYS,
    _canonicalize,
    _record_from_cache,
)


#: The real `bookSection` payload — Elsevier, via Crossref.
CHAPTER_PAYLOAD = {
    "itemType": "bookSection",
    "title": "Collagen Fibril Assembly and Function",
    "bookTitle": "Current Topics in Developmental Biology",
    "publisher": "Elsevier",
    "volume": "130",
    "pages": "107-142",
    "date": "2018",
    "DOI": "10.1016/bs.ctdb.2018.02.004",
    "creators": [
        {"creatorType": "author", "firstName": "David F.", "lastName": "Holmes"},
        {"creatorType": "author", "firstName": "Yinhui", "lastName": "Lu"},
    ],
}

#: The real `journalArticle` payload — the NMR benchmark paper.
ARTICLE_PAYLOAD = {
    "itemType": "journalArticle",
    "title": "An NMR method to characterize multiple water compartments",
    "publicationTitle": "Cell Biology International",
    "date": "2006",
    "DOI": "10.1016/j.cellbi.2005.09.009",
    "creators": [
        {"creatorType": "author", "firstName": "Ping", "lastName": "Zhu"},
    ],
}


class TestTheChapterContainer:
    def test_the_book_title_is_no_longer_dropped(self):
        """THE DEFECT, directly."""
        assert _canonicalize(CHAPTER_PAYLOAD)["journal"] == \
            "Current Topics in Developmental Biology"

    def test_it_does_not_come_from_the_publisher_instead(self):
        """
        Elsevier is the publisher, not the container. Filling the container
        slot from the publisher would look like a fix and be wrong.
        """
        record = _canonicalize(CHAPTER_PAYLOAD)
        assert record["journal"] != record["publisher"]
        assert record["publisher"] == "Elsevier"

    def test_the_chapter_title_stays_the_chapter_title(self):
        """The container must not overwrite the document's own title."""
        assert _canonicalize(CHAPTER_PAYLOAD)["title"] == \
            "Collagen Fibril Assembly and Function"


class TestTheArticleStillWorks:
    """The old behaviour is the common case and must not move."""

    def test_publication_title_is_still_read(self):
        assert _canonicalize(ARTICLE_PAYLOAD)["journal"] == "Cell Biology International"

    def test_publication_title_wins_when_both_are_present(self):
        """
        Priority order matters: an item carrying both is a journal article
        that happens to record a book title, not a chapter.
        """
        both = dict(ARTICLE_PAYLOAD, bookTitle="Some Compendium")
        assert _canonicalize(both)["journal"] == "Cell Biology International"


class TestOtherItemTypes:
    @pytest.mark.parametrize("key,expected", [
        ("publicationTitle", "A Journal"),
        ("bookTitle", "A Volume"),
        ("proceedingsTitle", "A Conference"),
        ("encyclopediaTitle", "An Encyclopedia"),
        ("journal", "A Legacy Field"),
    ])
    def test_every_declared_key_is_actually_read(self, key, expected):
        """
        A key listed in CONTAINER_TITLE_KEYS but never consulted would be the
        same class of bug all over again.
        """
        assert _canonicalize({key: expected})["journal"] == expected

    def test_a_whole_book_has_no_container_and_that_is_correct(self):
        """
        A monograph IS the container. None is the right answer here — and it
        is falsy, so the completeness rule's `publisher|isbn` branch decides,
        which is what a book should be judged on.
        """
        book = {"itemType": "book", "title": "Progress in Heritable Soft "
                                             "Connective Tissue Diseases",
                "publisher": "Springer", "date": "2021"}
        assert _canonicalize(book)["journal"] is None


class TestItFailsSafely:
    def test_a_blank_container_reads_as_missing_not_as_empty_text(self):
        """
        An empty string is truthy enough to pass a `field in metadata` check
        while carrying nothing. It must degrade to None so guards fire.
        """
        assert _canonicalize({"publicationTitle": "   "})["journal"] is None

    def test_a_blank_value_falls_through_to_the_next_key(self):
        payload = {"publicationTitle": "", "bookTitle": "The Real Container"}
        assert _canonicalize(payload)["journal"] == "The Real Container"

    def test_a_non_string_container_is_ignored_rather_than_returned(self):
        """Malformed upstream data must not become a citation field."""
        assert _canonicalize({"publicationTitle": {"nested": "junk"}})["journal"] is None

    def test_surrounding_whitespace_is_trimmed(self):
        assert _canonicalize({"bookTitle": "  Padded Volume \n"})["journal"] == \
            "Padded Volume"

    def test_an_empty_payload_does_not_raise(self):
        assert _canonicalize({})["journal"] is None

    def test_a_non_dict_payload_does_not_raise(self):
        assert _canonicalize(None)["journal"] is None


class TestTheCacheDoesNotFreezeTheMapping:
    """
    A cache of DERIVED values silently outlives the code that derived them.

    This was not hypothetical: after `_canonicalize` learned to read
    `bookTitle`, a real re-run of the Elsevier chapter STILL produced
    `journal: None` — the value came from a cache file written by the previous
    version, and a cache hit returned it verbatim. The fix was live, the unit
    tests passed, and the pipeline's answer did not move.
    """

    def _entry(self, payload):
        """A cache entry as `zotero_cache/doi_*.json` actually stores one."""
        return {"record": dict(_canonicalize(payload)), "confidence": 0.95,
                "source": "zotero_api"}

    def test_a_cached_chapter_picks_up_the_new_mapping(self):
        entry = self._entry(CHAPTER_PAYLOAD)
        entry["record"]["journal"] = None          # what the old code wrote

        assert _record_from_cache(entry)["journal"] == \
            "Current Topics in Developmental Biology"

    def test_authors_survive_the_rebuild(self):
        """
        THE REASON THE OLD CODE REFUSED TO RE-DERIVE. It canonicalized the
        cache WRAPPER, which has no `creators`, so authors vanished. Deriving
        from the record's own payload keeps them.
        """
        record = _record_from_cache(self._entry(CHAPTER_PAYLOAD))

        assert record["authors"] == ["David F. Holmes", "Yinhui Lu"]

    def test_every_other_field_survives_too(self):
        record = _record_from_cache(self._entry(CHAPTER_PAYLOAD))

        assert record["title"] == "Collagen Fibril Assembly and Function"
        assert record["doi"] == "10.1016/bs.ctdb.2018.02.004"
        assert record["publisher"] == "Elsevier"
        assert record["year"] == "2018"

    def test_an_entry_with_no_payload_keeps_what_it_has(self):
        """
        Old cache files predate payloads being stored. A stale mapping still
        beats discarding the metadata entirely.
        """
        legacy = {"record": {"title": "Older Entry", "authors": ["A B"],
                             "journal": "Some Journal"}}

        assert _record_from_cache(legacy)["title"] == "Older Entry"
        assert _record_from_cache(legacy)["journal"] == "Some Journal"

    def test_an_empty_entry_does_not_raise(self):
        assert _record_from_cache({})["journal"] is None


class TestTheKeyListItself:
    def test_the_article_key_is_first(self):
        """Priority is load-bearing — see the both-present test above."""
        assert CONTAINER_TITLE_KEYS[0] == "publicationTitle"

    def test_the_chapter_key_is_present(self):
        assert "bookTitle" in CONTAINER_TITLE_KEYS
