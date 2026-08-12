#!/usr/bin/env python3
"""
Layout detection via a grounding-capable VLM — the accurate-but-slow fallback.

For biology researchers: some vision models can be asked to "draw boxes" around
the regions of a page (tables, figures, captions) instead of transcribing it.
V5 used DeepSeek-OCR-2 this way, and it found every table on the test pages.
The cost is what motivates V6: ~17 s of GPU time per page, against ~15 ms for
the ONNX detector. This implementation exists so that the config can always
fall back to the proven detector while the cheap one earns trust.

## Staying VLM-agnostic

This module carries no knowledge of which model produces the grounding text.
Everything model-specific arrives in a `GroundingSpec` built from the model's
profile YAML (`grounding_format` section): the box regex, the coordinate grid,
and the label vocabulary. The model call itself is a plain callable injected by
the caller — this module never talks to a provider directly, which also makes
it testable without a GPU.

Native labels are mapped onto the canonical vocabulary in `base.py` before
detections leave this module, so consumers never see model-specific label
strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from src.processing.layout.base import (
    LABEL_IMAGE,
    LABEL_IMAGE_CAPTION,
    LABEL_TABLE,
    LABEL_TABLE_CAPTION,
    LABEL_TABLE_FOOTNOTE,
    Detection,
    scale_normalized_box,
)
from src.processing.pdf.base import PdfPage


@dataclass
class GroundingSpec:
    """
    Everything model-specific about reading a layout VLM's grounding output.

    Sourced from the layout model's profile config so that a detector swap never
    touches code. Ported from V5's `table_region_detector.GroundingSpec`.

    Attributes:
        box_pattern: regex with named groups `label` and `coords` matching one
            labelled box. It must accept every surface form the model emits —
            DeepSeek produces two depending on how it is served:
                llama-mtmd-cli:  <|ref|>table<|/ref|><|det|>[[85, 101, 915, 786]]<|/det|>
                llama-server:    table[[85, 101, 915, 786]]
            A pattern written for one form silently matches nothing against the
            other, which looks exactly like a total detection failure.
        coordinate_scale: the grid the model normalizes boxes to (DeepSeek:
            1000), independent of render DPI.
        table_label / caption_label / footnote_label: the model's native names
            for a table body, table caption, and table footnote.
        image_label / image_caption_label: native names for figure body and
            figure caption. Optional — a detector without figure grounding
            leaves them empty, which turns figure support off.
        prompt: the grounding prompt to send the model, kept with the spec so
            the caller need not hardcode it.
    """

    box_pattern: str
    coordinate_scale: float
    table_label: str
    caption_label: str
    footnote_label: str
    image_label: str = ""
    image_caption_label: str = ""
    prompt: str = ""
    _compiled: "re.Pattern" = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        self._compiled = re.compile(self.box_pattern, re.MULTILINE)

    @classmethod
    def from_config(cls, grounding_format: dict) -> "GroundingSpec":
        """Build a spec from a profile's `grounding_format` config dict."""
        return cls(
            box_pattern=grounding_format["box_pattern"],
            coordinate_scale=float(grounding_format["coordinate_scale"]),
            table_label=grounding_format["table_label"],
            caption_label=grounding_format["caption_label"],
            footnote_label=grounding_format["footnote_label"],
            image_label=grounding_format.get("image_label", ""),
            image_caption_label=grounding_format.get("image_caption_label", ""),
            prompt=grounding_format.get("prompt", ""),
        )

    @property
    def supports_figures(self) -> bool:
        """True when this model declares a figure label."""
        return bool(self.image_label)

    def canonical_label_map(self) -> Dict[str, str]:
        """Native model labels -> the canonical vocabulary consumers expect."""
        mapping = {
            self.table_label: LABEL_TABLE,
            self.caption_label: LABEL_TABLE_CAPTION,
            self.footnote_label: LABEL_TABLE_FOOTNOTE,
        }
        if self.image_label:
            mapping[self.image_label] = LABEL_IMAGE
        if self.image_caption_label:
            mapping[self.image_caption_label] = LABEL_IMAGE_CAPTION
        return mapping


class VlmGroundingDetector:
    """
    `LayoutDetector` implementation backed by a grounding VLM call.

    The model call is injected as `run_grounding(page_image) -> str` so this
    class stays independent of the provider layer (and unit-testable with a
    canned response). In production the caller wires it to the layout
    provider's `read_crop(page_image, spec.prompt, image_first=True)`.
    """

    def __init__(self, spec: GroundingSpec,
                 run_grounding: Callable[["Image.Image"], str]):
        self._spec = spec
        self._run_grounding = run_grounding
        self._label_map = spec.canonical_label_map()

    @property
    def name(self) -> str:
        return "vlm"

    @property
    def supports_figures(self) -> bool:
        return self._spec.supports_figures

    def detect(self, page_image: "Image.Image",
               page: Optional[PdfPage] = None) -> List[Detection]:
        """
        Ask the VLM for grounding boxes and parse them into pixel detections.

        An empty or unparseable response yields an empty list — the caller's
        coverage validator is what decides whether that is a problem.
        """
        raw_text = self._run_grounding(page_image)
        if not raw_text:
            return []
        return self.parse(raw_text, page_image.size)

    def parse(self, raw_text: str,
              image_size: Tuple[int, int]) -> List[Detection]:
        """
        Parse grounding text into detections, without making a model call.

        Split out from `detect` so a stored grounding response (e.g. from a
        benchmark run) can be re-parsed offline.
        """
        detections: List[Detection] = []
        for match in self._spec._compiled.finditer(raw_text):
            numbers = [int(value) for value in match.group("coords").split(",")]
            if len(numbers) != 4:
                continue  # malformed box — skip it rather than guess
            native_label = match.group("label")
            detections.append(Detection(
                # Unknown native labels pass through unchanged; consumers
                # ignore labels they do not recognize.
                label=self._label_map.get(native_label, native_label),
                box_px=scale_normalized_box(
                    numbers, image_size, self._spec.coordinate_scale),
            ))
        return detections
