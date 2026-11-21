"""
Find each answer-key paragraph inside a tool's running text (protocol §6.2).

For biology researchers: tools split text into paragraphs differently (by
PDF line, by page, by detected block), so paragraphs cannot be compared one
to one. Instead each key paragraph is searched for in the tool's whole text,
like aligning a read to a genome: word 4-grams find the likely region
(seeding), then a fuzzy local alignment finds the exact span (extension).
The span's similarity to the key paragraph says whether it was read; its
position says in which order; and an in-text link is attributed to the key
paragraph whose span contains it.
"""
from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .text_utils import INVISIBLE


@dataclass
class Stream:
    """A tool's body as one normalised string, with a map back to
    (paragraph index, offset in that paragraph) for every character."""
    text: str
    origin: list[tuple[int, int]]
    paragraph_starts: list[int]

    def position_of(self, paragraph_index: int, offset: int) -> int | None:
        """Normalised position of a raw (paragraph, offset) location."""
        if paragraph_index >= len(self.paragraph_starts):
            return None
        start = self.paragraph_starts[paragraph_index]
        end = (self.paragraph_starts[paragraph_index + 1]
               if paragraph_index + 1 < len(self.paragraph_starts) else len(self.text))
        offsets = [self.origin[i][1] for i in range(start, end)]
        return start + bisect.bisect_left(offsets, offset) if offsets else start


@dataclass
class Span:
    key_index: int
    start: int
    end: int
    score: float          # similarity of key paragraph to the span, 0-1


def build_stream(paragraphs: list[str]) -> Stream:
    """Lowercase alphanumerics, everything else one space; keep the origin
    of each kept character so raw offsets can be mapped in."""
    chars, origin, starts = [], [], []
    for p_index, paragraph in enumerate(paragraphs):
        starts.append(len(chars))
        previous_space = True
        skip = line_end_hyphens(paragraph)
        for offset, char in enumerate(paragraph):
            if offset in skip:
                continue
            for folded in fold(char):
                if folded != " ":
                    chars.append(folded)
                    origin.append((p_index, offset))
                    previous_space = False
                elif not previous_space:
                    chars.append(" ")
                    origin.append((p_index, offset))
                    previous_space = True
        if chars and chars[-1] != " ":
            chars.append(" ")
            origin.append((p_index, len(paragraph)))
    return Stream("".join(chars), origin, starts)


LINE_END_HYPHEN = re.compile(r"(?<=\w)[-­‐\x02][ \t]*\r?\n[ \t]*(?=[a-z])")


def line_end_hyphens(text: str) -> set[int]:
    """Character positions to drop so that a word broken over a line end
    ('differ-' newline 'entiation') is joined again (§5 step 3)."""
    skip: set[int] = set()
    for found in LINE_END_HYPHEN.finditer(text):
        skip.update(range(found.start(), found.end()))
    return skip


def fold(char: str) -> str:
    """One character -> NFKC, lowercase, non-alphanumerics as a space;
    invisible characters (zero-width space, soft hyphen) vanish. Used for both
    the key and the tool text, so they are folded identically."""
    if char in INVISIBLE:
        return ""
    return "".join(c if c.isalnum() else " "
                   for c in unicodedata.normalize("NFKC", char).lower())


def normalise_paragraph(text: str) -> str:
    """A key paragraph folded exactly as build_stream folds tool text."""
    return re.sub(r" +", " ", "".join(fold(c) for c in text)).strip()


def align_paragraphs(key_paragraphs: list[str], stream: Stream) -> list[Span]:
    """One Span per key paragraph (score 0 and start -1 when not found)."""
    index4 = shingle_index(stream.text, 4)
    index2 = None
    spans = []
    for k, paragraph in enumerate(key_paragraphs):
        needle = normalise_paragraph(paragraph)
        if len(needle.split()) >= 6:
            spans.append(align_one(k, needle, stream.text, index4, 4))
        else:                       # short paragraph: seed with word pairs
            index2 = index2 or shingle_index(stream.text, 2)
            spans.append(align_one(k, needle, stream.text, index2, 2))
    return spans


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def shingles(words: list[str], n: int) -> list[str]:
    return [" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]


def shingle_index(text: str, n: int = 4) -> dict[str, list[int]]:
    """Word n-gram -> character positions where it starts in the stream."""
    index: dict[str, list[int]] = {}
    positions = [m.start() for m in re.finditer(r"\S+", text)]
    words = text.split()
    for i, gram in enumerate(shingles(words, n)):
        index.setdefault(gram, []).append(positions[i])
    return index


def align_one(k: int, needle: str, haystack: str, index: dict, n: int) -> Span:
    if not needle or not haystack:
        return Span(k, -1, -1, 0.0)
    hits = sorted(p for gram in set(shingles(needle.split(), n))
                  for p in index.get(gram, [])[:50])
    if not hits:
        return Span(k, -1, -1, 0.0)
    window = densest_window(hits, int(len(needle) * 1.3) + 40)
    lo = max(0, window - len(needle) // 3 - 40)
    hi = min(len(haystack), window + int(len(needle) * 1.5) + 80)
    region = haystack[lo:hi]
    alignment = fuzz.partial_ratio_alignment(needle, region)
    start, end = lo + alignment.dest_start, lo + alignment.dest_end
    score = Levenshtein.normalized_similarity(needle, haystack[start:end])
    return Span(k, start, end, score)


def densest_window(sorted_positions: list[int], width: int) -> int:
    """Start of the width-wide window holding the most seed hits."""
    best_start, best_count, left = sorted_positions[0], 0, 0
    for right, position in enumerate(sorted_positions):
        while position - sorted_positions[left] > width:
            left += 1
        if right - left + 1 > best_count:
            best_count, best_start = right - left + 1, sorted_positions[left]
    return best_start
