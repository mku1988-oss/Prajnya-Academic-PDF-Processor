#!/usr/bin/env python3
"""
A numpy scalar must never reach the output JSON — and must never destroy it.

Found on the full sample run of session 9, in the most expensive way possible:
a 25-page paper made every one of its GPU calls, finished, and then died at the
last step with

    PydanticSerializationError: Unable to serialize unknown type: <class 'numpy.intc'>

Eight minutes of GPU work, discarded while writing a file.

The leak is `find_unclaimed_ink`: OpenCV's `connectedComponentsWithStats`
returns a numpy array, so unpacking a row yields `numpy.intc`. Those values
become a table's box, the box goes into the page metadata, and the metadata is
serialized. Nothing in between notices, because numpy scalars do arithmetic and
compare exactly like ints.

Two tests, because there are two distinct defects and fixing either one alone
leaves the other:

  1. the boxes are plain ints          — fix it where it starts
  2. writing survives one anyway       — never lose a finished document again
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.processing.layout.regions import find_unclaimed_ink


class TestTheLeakAtItsSource:
    def make_ink(self) -> np.ndarray:
        """A page with one blob of ink big enough to count as content."""
        ink = np.zeros((200, 200), dtype=bool)
        ink[120:150, 20:180] = True          # a wide band, like a footnote line
        return ink

    def test_orphan_boxes_are_plain_python_ints(self):
        blobs = find_unclaimed_ink(self.make_ink(), claimed_boxes=[])

        assert blobs, "the test's own ink blob was not detected"
        for box in blobs:
            for value in box:
                assert type(value) is int, \
                    f"{value!r} is {type(value).__name__}, not int"

    def test_the_boxes_are_json_serializable(self):
        """The property that actually matters, stated directly."""
        blobs = find_unclaimed_ink(self.make_ink(), claimed_boxes=[])
        assert json.dumps(blobs)

    def test_a_numpy_int_is_not_json_serializable(self):
        """Why this test file exists: numpy ints look like ints until they don't."""
        value = np.intc(5)

        assert value == 5 and value + 1 == 6      # behaves like an int...
        with pytest.raises(TypeError):            # ...until it is written out
            json.dumps({"x": value})


class TestWritingSurvivesOneAnyway:
    def processed_profile(self, tmp_path: Path, page_metadata):
        from src.processing.models import (
            ContentPage, DocumentProfile, ExtractedMetadata, ProcessingResult,
        )

        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        return DocumentProfile(
            pdf_path=pdf, page_count=1, file_size=pdf.stat().st_size,
            processing_result=ProcessingResult(
                metadata=ExtractedMetadata(title="A paper"),
                pages=[ContentPage(page_number=1, content="body text",
                                   metadata=page_metadata)]))

    def test_a_leaked_numpy_scalar_no_longer_costs_the_document(self, tmp_path, caplog):
        """The whole point: one bad field, not one lost run."""
        import logging

        from run_prajnya import _write_result

        profile = self.processed_profile(
            tmp_path, {"table_boxes": [[np.intc(10), np.intc(20),
                                        np.intc(30), np.intc(40)]]})
        out_path = tmp_path / "paper.v6.json"

        with caplog.at_level(logging.WARNING):
            _write_result(profile, out_path, elapsed=171.0)

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["pages"][0]["content"] == "body text"
        assert payload["metadata_final"]["title"] == "A paper"
        assert payload["pages"][0]["metadata"]["table_boxes"] == [[10, 20, 30, 40]]
        assert any("fix this at its source" in message for message in caplog.messages)

    def test_an_ordinary_document_is_written_exactly_as_before(self, tmp_path):
        """The fallback must not change the shape of a clean run."""
        from run_prajnya import _write_result

        profile = self.processed_profile(tmp_path, {"figure_count": 2})
        out_path = tmp_path / "paper.v6.json"
        _write_result(profile, out_path, elapsed=12.5)

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["document_info"]["processing_time"] == 12.5
        assert payload["summary"]["figure_count"] == 2
        # Contract 1.3 adds the furniture and outside-body labels (none here) and `body`.
        assert payload["pages"][0]["metadata"] == {"figure_count": 2, "furniture": [], "outside_body": []}
