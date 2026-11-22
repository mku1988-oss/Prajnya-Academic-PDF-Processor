#!/usr/bin/env python3
"""
`--force` means redo the work, and metadata is work.

For biology researchers: the ledger remembers what a document's metadata was
last time, so a second run does not repeat the Zotero lookup. That is a good
default and it stays the default. But it meant `--force` — the flag whose
whole purpose is "process this again" — re-read every page of the PDF and then
returned the metadata row written by the FIRST run.

THE CONSEQUENCE, which cost two full re-runs to find: a fixed extractor could
not correct a stored mistake. When the Zotero mapping was taught to read a
book chapter's `bookTitle`, the fix was live, its unit tests were green, and
two live re-runs still produced `journal: None` — because the value never came
from the extractor at all. Nothing failed; nothing changed; there was no
symptom to chase.

This is the third cache in one session to hide a fix behind a stored derived
value (the other two are in `test_container_title.py`). The pattern is worth
naming: **a cache of derived values silently outlives the code that derived
them.**

NOTE ON WHAT IS *NOT* PROTECTED HERE. An earlier design preserved rows a human
had corrected via `--review`, on the grounds that a machine re-run should not
discard a person's answer. That was dropped deliberately: V6's metadata is
good enough that the review flow no longer fires — the last full corpus run
flagged **0 of 9 documents** — so the special case protected a path that never
executes, at the cost of making `--force` mean two different things. If review
becomes load-bearing again, this is the test to revisit.
"""

import logging

import pytest

from src.processing.ledger.semantic_ledger import LedgerRecord, generate_hash
from src.processing.strategies.research_article.metadata_enrichment import (
    enrich_metadata,
)

@pytest.fixture
def pdf(tmp_path):
    """A real file on disk: the ledger's document id is a hash of its bytes."""
    path = tmp_path / "example.pdf"
    path.write_bytes(b"%PDF-1.4 not really a pdf, but it has bytes to hash")
    return str(path)


class FakeLedger:
    """Just enough ledger to answer `get_record`, and to say if it was asked."""

    def __init__(self, record=None):
        self.record = record
        self.lookups = 0

    def get_record(self, document_id):
        self.lookups += 1
        return self.record


class FakeZotero:
    """Records whether it was consulted — the proof that work was redone."""

    def __init__(self, journal="Current Topics in Developmental Biology"):
        self.calls = 0
        self.journal = journal

    def get_canonical_metadata(self, vlm_metadata):
        self.calls += 1
        return {
            "record": {"title": "Collagen Fibril Assembly and Function",
                       "authors": ["David F. Holmes"], "doi": "10.1016/x",
                       "journal": self.journal, "publisher": "Elsevier",
                       "year": "2018"},
            "confidence": 0.95,
            "source": "zotero_api",
        }


@pytest.fixture
def stale_row(pdf):
    """A row from an earlier run, written before the mapping was fixed."""
    return LedgerRecord(
        document_id=generate_hash(pdf),
        pdf_filename="example.pdf",
        source_hash=generate_hash(pdf),
        last_updated="2026-08-07T00:00:00Z",
        title="Collagen Fibril Assembly and Function",
        authors=["David F. Holmes"],
        doi="10.1016/x",
        journal=None,                      # THE STORED MISTAKE
        year="2018",
        validation_confidence=0.95,
        validation_source="zotero_api",
    )


@pytest.fixture
def hypothesis():
    return {"title": "Collagen Fibril Assembly and Function", "doi": "10.1016/x",
            "authors": ["David F. Holmes"], "year": "2018"}


class TestTheDefaultIsUnchanged:
    def test_a_stored_row_is_reused_and_zotero_is_not_called(self, pdf, stale_row,
                                                             hypothesis):
        """The cache is a real saving on an ordinary re-run. Keep it."""
        ledger, zotero = FakeLedger(stale_row), FakeZotero()

        result = enrich_metadata(hypothesis, pdf, ledger=ledger,
                                 zotero_service=zotero,
                                 logger=logging.getLogger("test"))

        assert zotero.calls == 0
        assert result["journal"] is None       # the stored value, warts and all


class TestForceRederives:
    def test_the_stored_row_no_longer_wins(self, pdf, stale_row, hypothesis):
        """THE DEFECT: a corrected mapping must be able to change the answer."""
        ledger, zotero = FakeLedger(stale_row), FakeZotero()

        result = enrich_metadata(hypothesis, pdf, ledger=ledger,
                                 zotero_service=zotero, use_ledger_cache=False,
                                 logger=logging.getLogger("test"))

        assert zotero.calls == 1
        assert result["journal"] == "Current Topics in Developmental Biology"

    def test_the_ledger_is_not_even_consulted(self, pdf, stale_row, hypothesis):
        ledger = FakeLedger(stale_row)

        enrich_metadata(hypothesis, pdf, ledger=ledger, zotero_service=FakeZotero(),
                        use_ledger_cache=False, logger=logging.getLogger("test"))

        assert ledger.lookups == 0

    def test_provenance_is_rebuilt_rather_than_inherited(self, pdf, stale_row,
                                                         hypothesis):
        """A re-derived record must not carry the old run's provenance."""
        result = enrich_metadata(hypothesis, pdf, ledger=FakeLedger(stale_row),
                                 zotero_service=FakeZotero(),
                                 use_ledger_cache=False,
                                 logger=logging.getLogger("test"))

        assert result["validation_source"] == "zotero_api"
        assert "provenance" in result

    def test_a_document_with_no_stored_row_behaves_the_same_either_way(
            self, pdf, hypothesis):
        both = [enrich_metadata(dict(hypothesis), pdf, ledger=FakeLedger(None),
                                zotero_service=FakeZotero(),
                                use_ledger_cache=flag,
                                logger=logging.getLogger("test"))["journal"]
                for flag in (True, False)]

        assert both[0] == both[1] == "Current Topics in Developmental Biology"


class TestTheFlagReachesTheStrategies:
    """
    A parameter nothing passes is decoration — and this exact class of gap
    (code written, tested, never wired) shipped once already this session.
    """

    def test_the_router_accepts_it(self):
        import inspect

        from src.processing.strategies.router import build_strategy

        assert "refresh_metadata" in inspect.signature(build_strategy).parameters

    def test_the_article_strategy_passes_it_to_enrichment(self):
        import inspect

        from src.processing.strategies.research_article.strategy import (
            ResearchArticleStrategy,
        )

        source = inspect.getsource(ResearchArticleStrategy._enrich)
        assert "use_ledger_cache=not self.refresh_metadata" in source

    def test_the_runner_ties_it_to_force(self):
        """
        `--force` reaches the strategy through ONE function now.

        This used to be asserted twice, against `run_one_document` and
        `_process_entries` in the entrypoint — the two places that each built a
        strategy. They are now one (`runner.process_one`), which is why a
        single assertion is the honest one: there is no second code path left
        to drift.
        """
        import inspect

        from src.processing.runner import process_one

        assert "refresh_metadata=request.force" in inspect.getsource(process_one)

    def test_the_command_line_flag_reaches_the_request(self):
        """The other half: `--force` on the command line becomes `request.force`."""
        from argparse import Namespace

        from src.processing.run_request import RunRequest

        args = Namespace(pdf="paper.pdf", out=None, backend=None, pages=None,
                         chapters=None, max_pages=None, front_matter_pages=10,
                         force=True, recursive=False, noninteractive=False,
                         verbose=False)
        assert RunRequest.from_args(args).force is True
