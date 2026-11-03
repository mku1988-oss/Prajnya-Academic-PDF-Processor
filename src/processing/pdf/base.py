#!/usr/bin/env python3
"""
The PDF access interface — one seam for all text, geometry, and rendering.

For biology researchers: every part of the pipeline that needs to look inside a
PDF goes through this interface rather than calling a PDF library directly. Two
reasons:

1. **Licensing.** PyMuPDF (the fast default) is AGPL-3.0. Routing everything
   through here means swapping in a permissively-licensed library is a config
   change, not a rewrite. See CLAUDE.md, "The Licence Rule".
2. **Correctness.** PDF libraries disagree about which corner of the page is the
   origin. Normalizing that in one place — here — stops a whole category of
   silent bugs where a figure gets cropped from the wrong part of the page.

## Coordinate convention (important)

**Everything this interface returns uses a top-left origin, y increasing
downward, in PDF points.**

That matches PyMuPDF and the way images are addressed, so page-image pixel
coordinates and PDF coordinates differ only by a scale factor. PDFium natively
uses a *bottom-left* origin with y increasing upward, so its backend flips the
y-axis on the way out. Implementations must do this conversion themselves; no
caller should ever have to think about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Protocol, Sequence, runtime_checkable

try:  # Pillow is only needed for rendering; keep the import soft for type-checking.
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rect:
    """
    A rectangle in PDF points, top-left origin, y increasing downward.

    `x0, y0` is the top-left corner and `x1, y1` the bottom-right, so `y0 < y1`
    always holds for a well-formed rectangle.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def normalized(self) -> "Rect":
        """Return a copy with the corners ordered so x0<=x1 and y0<=y1."""
        return Rect(
            min(self.x0, self.x1), min(self.y0, self.y1),
            max(self.x0, self.x1), max(self.y0, self.y1),
        )

    def scaled(self, factor: float) -> "Rect":
        """Scale into another coordinate space — e.g. PDF points to image pixels."""
        return Rect(self.x0 * factor, self.y0 * factor, self.x1 * factor, self.y1 * factor)

    def intersection(self, other: "Rect") -> "Rect":
        """Overlapping region; a zero-area rect when they do not overlap."""
        return Rect(
            max(self.x0, other.x0), max(self.y0, other.y0),
            min(self.x1, other.x1), min(self.y1, other.y1),
        ).clipped()

    def clipped(self) -> "Rect":
        """Collapse a negative-extent rect to zero area rather than letting it invert."""
        return Rect(self.x0, self.y0, max(self.x0, self.x1), max(self.y0, self.y1))

    def iou(self, other: "Rect") -> float:
        """
        Intersection-over-union, the standard box-agreement measure.

        Used to check that a swapped-in layout detector finds the same regions as
        the one it replaces (see the Stage 2 gate in plan.md).
        """
        overlap = self.intersection(other).area
        union = self.area + other.area - overlap
        return overlap / union if union > 0 else 0.0

    def contains_center_of(self, other: "Rect") -> bool:
        """
        Whether `other`'s midpoint falls inside this rect.

        More forgiving than full containment, which matters when matching a word
        to a region whose box was drawn slightly tight.
        """
        cx, cy = (other.x0 + other.x1) / 2, (other.y0 + other.y1) / 2
        return self.x0 <= cx <= self.x1 and self.y0 <= cy <= self.y1


@dataclass(frozen=True)
class Word:
    """A single word with its bounding box, in the interface's coordinates."""

    text: str
    rect: Rect


@dataclass(frozen=True)
class StyledLine:
    """A typeset line, with the typography needed to tell a heading from prose.

    For biology researchers: a section heading is printed differently from
    body text -- usually bolder, often larger, frequently in capitals. Those
    three facts are what this carries, because the WORDS alone cannot say
    whether "Introduction" is a top-level heading or a sub-heading.

    Attributes:
        text: The line as printed.
        rect: Its bounding box.
        cap_height: Median glyph height of the CAPITAL letters and digits on
            the line. Capitals are used deliberately: they have no descenders,
            so an ALL-CAPS heading stays comparable with mixed-case body text.
            Measuring whole-word boxes instead makes caps headings measure
            SMALLER than prose, which is backwards -- see the note in
            `utils/heading_pass.py`.
        weight: Median font weight (PDF units, ~400 regular, ~700 bold), or
            0.0 when the file does not record it. Some PDFs genuinely do not,
            so a consumer must treat 0.0 as "unknown", never as "light".
        caps_fraction: Share of the line's letters that are uppercase.
    """

    text: str
    rect: Rect
    cap_height: float = 0.0
    weight: float = 0.0
    caps_fraction: float = 0.0


@dataclass(frozen=True)
class ImageBlock:
    """A raster image embedded in the page — a figure candidate."""

    rect: Rect


@dataclass(frozen=True)
class DrawingBlock:
    """
    A vector drawing region — charts, rules, and diagrams.

    Distinguished from `ImageBlock` because born-digital figures are often vector
    art rather than raster images, and a page full of table rules looks very
    different from a page with one embedded photograph.
    """

    rect: Rect


# ---------------------------------------------------------------------------
# Page and document protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class PdfPage(Protocol):
    """One page. Coordinates follow the module-level convention."""

    @property
    def number(self) -> int:
        """Zero-based page index."""
        ...

    @property
    def size(self) -> tuple[float, float]:
        """Page (width, height) in PDF points."""
        ...

    def text(self) -> str:
        """Full text layer in reading order as the library reports it."""
        ...

    def words(self) -> List[Word]:
        """Every word with its box. The basis of the word-coverage check."""
        ...

    def text_in(self, rect: Rect) -> str:
        """
        Text whose words fall inside `rect`.

        This is how figure captions are read for free on born-digital pages,
        instead of paying a VLM call to transcribe them.
        """
        ...

    def styled_lines(self) -> List[StyledLine]:
        """
        Lines with their typography, for telling headings from prose.

        Separate from `words()` because the two are built differently: word
        boxes come from the library's own word or rect grouping, while these
        need per-character font data that only a character walk exposes. A
        page with no text layer returns an empty list rather than raising --
        a scan has no typography to report, and that must never cost the page
        its content.
        """
        ...

    def images(self) -> List[ImageBlock]:
        """Embedded raster images."""
        ...

    def drawings(self) -> List[DrawingBlock]:
        """Vector drawing regions."""
        ...

    def render(self, *, dpi: Optional[int] = None, max_edge: Optional[int] = None) -> "Image.Image":
        """
        Rasterize the page.

        Exactly one of `dpi` or `max_edge` should be given: `dpi` for a fixed
        resolution, `max_edge` to fit the longest side to a pixel budget (what
        the VLM profiles specify).
        """
        ...


@runtime_checkable
class PdfDocument(Protocol):
    """
    An open PDF.

    Use as a context manager so the handle is opened once per document rather
    than once per operation — V5 reopened the file three times per page, which
    this interface exists partly to stop.
    """

    @property
    def page_count(self) -> int:
        ...

    def page(self, index: int) -> PdfPage:
        ...

    def pages(self) -> Iterator[PdfPage]:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> "PdfDocument":
        ...

    def __exit__(self, *exc: object) -> None:
        ...


@runtime_checkable
class PdfBackend(Protocol):
    """A PDF library binding. Implementations live alongside this module."""

    @property
    def name(self) -> str:
        """Short identifier used in config and logs, e.g. "pymupdf"."""
        ...

    @property
    def licence(self) -> str:
        """SPDX identifier — surfaced so a licence-incompatible default is obvious."""
        ...

    def open(self, path: Path | str) -> PdfDocument:
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def flip_y(rect: Rect, page_height: float) -> Rect:
    """
    Convert between bottom-left and top-left origin conventions.

    The operation is its own inverse. PDFium reports a bottom-left origin, so its
    backend passes every rectangle through this on the way out.
    """
    return Rect(rect.x0, page_height - rect.y1, rect.x1, page_height - rect.y0)


def render_scale(page_size: tuple[float, float], *, dpi: Optional[int],
                 max_edge: Optional[int]) -> float:
    """
    Work out the points-to-pixels scale factor for a render request.

    Raises ValueError if neither or both of `dpi`/`max_edge` are supplied, since
    silently picking one would make render sizes unpredictable across backends.
    """
    if (dpi is None) == (max_edge is None):
        raise ValueError("render() needs exactly one of dpi or max_edge")
    if dpi is not None:
        return dpi / 72.0  # PDF user space is 72 points per inch
    return max_edge / max(page_size)


def words_in_rect(words: Sequence[Word], rect: Rect) -> List[Word]:
    """Select the words whose centers fall inside `rect`, preserving order."""
    return [w for w in words if rect.contains_center_of(w.rect)]


def looks_horizontal(word: Word) -> bool:
    """
    Is this word typeset left-to-right, rather than rotated up the margin?

    Neither backend exposes writing direction through this interface, but the
    box shape gives it away: a horizontal word of two or more characters is
    wider than it is tall, while a word rotated 90° (the "Downloaded from
    https://..." stamp many journals print down the outer margin) is taller
    than it is wide. Single-character words are near-square either way, so they
    get the benefit of the doubt — downstream tokenizers ignore them anyway.
    """
    if len(word.text.strip()) < 2:
        return True
    return word.rect.width >= word.rect.height


def group_words_into_lines(
    words: Sequence[Word],
    *,
    column_gap_factor: float = 1.5,
) -> List[List[Word]]:
    """
    Rebuild typeset lines from a flat word list.

    The backends return words without line structure, but two consumers need
    lines back: the content-drop check (to rejoin words hyphenated across line
    breaks) and the page classifier (to test whether a line looks like a
    centered equation). Reconstruction is geometric:

    1. Rotated marginalia are dropped (see `looks_horizontal`) — they are not
       part of the article and would poison both consumers.
    2. Words whose boxes overlap vertically by at least half the smaller
       height share a line.
    3. A horizontal gap wider than `column_gap_factor` × the word height
       starts a new line, so the two columns of a journal page do not fuse
       into one line spanning the gutter (word gaps are ~0.3× the height;
       column gutters are ~1.5–2×).

    Returns:
        Lines in top-to-bottom order, each line's words left-to-right.
    """
    horizontal = [w for w in words if looks_horizontal(w)]
    if not horizontal:
        return []

    # Pass 1: group by vertical overlap into y-bands.
    ordered = sorted(horizontal, key=lambda w: ((w.rect.y0 + w.rect.y1) / 2, w.rect.x0))
    bands: List[List[Word]] = []
    band_top, band_bottom = 0.0, 0.0
    for word in ordered:
        rect = word.rect
        overlap = min(band_bottom, rect.y1) - max(band_top, rect.y0)
        min_height = min(band_bottom - band_top, rect.height)
        if bands and min_height > 0 and overlap >= 0.5 * min_height:
            bands[-1].append(word)
            band_top = min(band_top, rect.y0)
            band_bottom = max(band_bottom, rect.y1)
        else:
            bands.append([word])
            band_top, band_bottom = rect.y0, rect.y1

    # Pass 2: split each band at column-sized horizontal gaps.
    lines: List[List[Word]] = []
    for band in bands:
        band.sort(key=lambda w: w.rect.x0)
        current = [band[0]]
        for word in band[1:]:
            previous = current[-1].rect
            gap = word.rect.x0 - previous.x1
            if gap > column_gap_factor * max(previous.height, word.rect.height):
                lines.append(current)
                current = [word]
            else:
                current.append(word)
        lines.append(current)
    return lines


def line_text(line: Sequence[Word]) -> str:
    """The text of one reconstructed line, words joined by single spaces."""
    return " ".join(w.text for w in line)


def line_rect(line: Sequence[Word]) -> Rect:
    """The union box of one reconstructed line."""
    return Rect(
        min(w.rect.x0 for w in line), min(w.rect.y0 for w in line),
        max(w.rect.x1 for w in line), max(w.rect.y1 for w in line),
    )


def coverage(extracted_text: str, reference_words: Sequence[Word]) -> float:
    """
    Fraction of the text layer's words that survived into `extracted_text`.

    This is the pipeline's content-loss alarm: if a VLM read a page and dropped a
    column, coverage falls and the page gets escalated or repaired. Comparison is
    on lowercased alphabetic tokens, so numbers and punctuation differences do
    not mask a real loss of prose.

    Returns 1.0 when the reference has no comparable words, so a blank page never
    looks like a failure.
    """
    import re  # local: keeps the module's import surface small

    def tokens(text: str) -> List[str]:
        return [t for t in re.findall(r"[a-z]+", text.lower()) if len(t) > 1]

    reference = tokens(" ".join(w.text for w in reference_words))
    if not reference:
        return 1.0

    found = set(tokens(extracted_text))
    return sum(1 for t in reference if t in found) / len(reference)


# ---------------------------------------------------------------------------
# Typography helpers -- SHARED so both backends compute the same quantity
# ---------------------------------------------------------------------------

#: Glyphs with a flat baseline and no descender. Cap-height is measured from
#: these alone, which is what makes an ALL-CAPS heading comparable with
#: mixed-case body text instead of measuring shorter than it.
CAP_GLYPHS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

#: Characters closer than this share a baseline, in PDF points.
BASELINE_TOLERANCE = 2.0

#: A gap wider than this many times the cap-height starts a new line, so two
#: columns of a journal page do not fuse across the gutter.
LINE_GAP_FACTOR = 3.0


def group_characters_into_lines(chars: List[dict]) -> List[List[dict]]:
    """
    Group a page's characters into typeset lines.

    Args:
        chars: dicts with `ch`, `x0`, `x1`, `y0`, `y1`, `weight`, in a
            bottom-up coordinate system.

    Returns:
        Lines top-to-bottom, each line's characters left-to-right, split
        wherever a horizontal gap is wide enough to be a column boundary.
    """
    if not chars:
        return []

    rows: List[List[dict]] = []
    current: List[dict] = []
    baseline: Optional[float] = None
    for char in sorted(chars, key=lambda c: (-c["y0"], c["x0"])):
        if baseline is None or abs(char["y0"] - baseline) <= BASELINE_TOLERANCE:
            current.append(char)
            baseline = char["y0"] if baseline is None else baseline
        else:
            rows.append(current)
            current, baseline = [char], char["y0"]
    if current:
        rows.append(current)

    lines: List[List[dict]] = []
    for row in rows:
        row = sorted(row, key=lambda c: c["x0"])
        heights = [c["y1"] - c["y0"] for c in row if c["y1"] > c["y0"]]
        typical = max(heights) if heights else 1.0
        run = [row[0]]
        for previous, char in zip(row, row[1:]):
            if char["x0"] - previous["x1"] > typical * LINE_GAP_FACTOR:
                lines.append(run)
                run = [char]
            else:
                run.append(char)
        lines.append(run)
    return lines


def styled_line_from_characters(
    chars: List[dict], page_height: float
) -> Optional["StyledLine"]:
    """
    Measure one line's typography.

    Args:
        chars: the line's characters, left-to-right, bottom-up coordinates.
        page_height: for flipping into the interface's top-left convention.

    Returns:
        A StyledLine, or None if the run holds no printable text.
    """
    import statistics

    if not chars:
        return None
    text = "".join(c["ch"] for c in chars).strip()
    if not text:
        return None

    caps = [c["y1"] - c["y0"] for c in chars
            if c["ch"] in CAP_GLYPHS and c["y1"] > c["y0"]]
    cap_height = statistics.median(caps) if caps else 0.0

    weights = [c["weight"] for c in chars if c.get("weight")]
    weight = statistics.median(weights) if weights else 0.0

    letters = [c["ch"] for c in chars if c["ch"].isalpha()]
    caps_fraction = (
        sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
    )

    box = Rect(
        min(c["x0"] for c in chars), min(c["y0"] for c in chars),
        max(c["x1"] for c in chars), max(c["y1"] for c in chars),
    )
    return StyledLine(
        text=text,
        rect=flip_y(box, page_height).normalized(),
        cap_height=round(cap_height, 3),
        weight=round(weight, 1),
        caps_fraction=round(caps_fraction, 3),
    )
