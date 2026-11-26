#!/usr/bin/env python3
"""
Superscripts and subscripts in a page's text layer, written as LaTeX (T37b).

For biology researchers: on a title page, each author's affiliation letter is
printed small and raised after the name ("Harvey-Samuel" + raised "a"). In
the PDF's stored text the letter is just another character with no space
before it, so the plain text reads "Harvey-Samuela" and a reader (human or
model) cannot tell the letter from the name. This module finds such
characters from their size and position and writes them the way the rest of
Prajnya's output already writes scripts: "Harvey-Samuel$^{a}$", "CO$_{2}$".

A character is a script when, compared with the last normal-sized character
before it ON THE SAME LINE, it is:
    - smaller: under 0.8 of that letter's height (T38b's G4 threshold), and
    - off its baseline by more than 0.15 of that height (G4 again), but by
      less than 0.7 of it: a bigger jump is the next line, not a script.
Raised = superscript `$^{...}$`, lowered = subscript `$_{...}$`.

The stored text and the character list come from the same PDF text order
but the list skips whitespace; they are walked together. If they do not line
up character for character, the plain text is returned unchanged and the
caller is told: a mis-aligned marking would put a script on the wrong letter.

Gate: eval/title_text_layer/GATE_T37b.md.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .text_first_page import BASELINE_SHIFT, SCRIPT_HEIGHT

#: A baseline shift at least this share of the letter's height is a new line.
NEXT_LINE_SHIFT = 0.7

#: How far left of the previous letter's right edge a script may start (in
#: that letter's heights): kerning can tuck a superscript slightly back.
TUCK_BACK = 0.3

SUPERSCRIPT = "^"
SUBSCRIPT = "_"
#: A prime drawn with the code of "0" by a maths font (T37c fix 3).
PRIME = "prime"


@dataclass
class MarkedText:
    """
    A page's text with its scripts marked.

    Attributes:
        text: the text; equal to the plain text layer when not `aligned`.
        aligned: the text and the characters lined up, so marking was done.
        runs: how many script groups were written.
    """
    text: str
    aligned: bool
    runs: int


def text_with_scripts(page) -> MarkedText:
    """
    The page's text layer with sub/superscripts written as `$^{..}$` / `$_{..}$`.

    Args:
        page: a `PdfPage` (its `text()` and `characters()`).

    Returns:
        A `MarkedText`. When the text and characters do not align, `text` is
        the plain text layer and `aligned` is False.
    """
    raw = page.text() or ""
    characters = page.characters()
    mapping, aligned = align(raw, characters)
    if not aligned:
        return MarkedText(text=raw, aligned=False, runs=0)
    kinds = script_kinds(characters)
    text, runs = write_marked(raw, mapping, kinds)
    return MarkedText(text=text, aligned=True, runs=runs)


def without_script_marks(value):
    """
    An author name (or list of names) with every `$^{..}$` / `$_{..}$` removed.

    For biology researchers: shot 2 reads the marked title-page text and
    copies the marks into the names it returns ("Mattia Serra$^{1}$*"). An
    affiliation mark is never part of a name, so the marks go; the name's
    letters stay. Lists keep their order; anything else is returned as is.
    """
    if isinstance(value, list):
        return [without_script_marks(item) for item in value]
    if not isinstance(value, str):
        return value
    text = _SCRIPT_GROUP.sub(" ", value)
    text = _BARE_SCRIPT.sub(" ", text)
    text = text.replace("$", " ")                      # a lone leftover sign
    # T37c fix 4: an ORCID icon stored as "ID" right after the surname
    # ("MiuraID"), and symbol characters ("☯", Unicode category So).
    text = _GLUED_ORCID.sub("", text)
    text = "".join(" " if unicodedata.category(c) == "So" else c for c in text)
    text = re.sub(r"\s+([,;])", r"\1", text)
    return " ".join(text.split())


def unicode_scripts(title):
    """
    A title with its `$_{..}$` / `$^{..}$` groups written in Unicode (T37d fix 1).

    For biology researchers: shot 2 copies the text's script marks into the
    title ("CO$_{2}$"). A bibliographic title must not carry LaTeX, so each
    group becomes Unicode subscripts or superscripts when every character in
    it has one ("CO₂", "Ca²⁺", "m⁶A"), and its plain characters otherwise
    ("T$_{reg}$" -> "Treg"). Anything that is not a string is returned as is.

    T37e: the models do not always write `$^{..}$`, so every notation seen in
    a title is read the same way: `$^{2+}$`, bare `^{2+}` / `_{12}` (the
    vision model's own titles: "Ca^{2+}", "P2RY_{12}"), `$^a$`, and the
    `^+^` form shot 2 once wrote ("CD8^+^HLA-DR^+^").
    """
    if not isinstance(title, str):
        return title

    def replace(match: "re.Match") -> str:
        table = _SUBSCRIPTS if match.group(1) == "_" else _SUPERSCRIPTS
        inner = match.group(2)
        if inner and all(c in table for c in inner):
            return "".join(table[c] for c in inner)
        return inner

    text = title
    for notation in _TITLE_NOTATIONS:
        text = notation.sub(replace, text)
    return " ".join(text.replace("$", "").split())


def recase_capitals_journal(journal):
    """
    A journal name copied from an all-capitals running header, re-cased (T37d fix 2).

    Only a name entirely in capitals with a word of five or more letters is
    touched: those words go to title case, "of / and / the / in / for / on" to
    lower case, and words of two to four letters stay in capitals (they are
    usually initialisms): "SCIENCE ADVANCES" -> "Science Advances", while
    "PNAS" and "PLOS ONE" are unchanged.
    """
    if not isinstance(journal, str) or journal != journal.upper():
        return journal
    words = journal.split()
    if not any(len(word) >= 5 and word.isalpha() for word in words):
        return journal
    recased = []
    for position, word in enumerate(words):
        if position > 0 and word.lower() in _SMALL_WORDS:
            recased.append(word.lower())
        elif len(word) >= 5:
            recased.append(word[0] + word[1:].lower())
        else:
            recased.append(word)
    return " ".join(recased)


#: The script notations read in a title, in this order, each capturing its
#: kind ("_"/"^") and contents (T37d fix 1, T37e):
#:   `$^{2+}$`, or bare `^{2+}` / `_{12}`: either `$` may be missing;
#:   `$^a$` / `$_2$`: a group without braces, between dollar signs;
#:   `^+^`: the superscript form shot 2 wrote once (no space inside).
_TITLE_NOTATIONS = (
    re.compile(r"\$?([\^_])\{([^{}]*)\}\$?"),
    re.compile(r"\$([\^_])([^$\s{}]{1,12})\$"),
    re.compile(r"(\^)([^\^\s{}$]{1,12})\^"),
)
_SUBSCRIPTS = dict(zip("0123456789+-−=()aehijklmnoprstuvx",
                       "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))
_SUPERSCRIPTS = dict(zip("0123456789+-−=()in",
                         "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁱⁿ"))
_SMALL_WORDS = {"of", "and", "the", "in", "for", "on"}


#: `$^{a,b}$`, `$_{2}$`, `$^{\text{}}$`: one script group, braces balanced
#: one level deep (LaTeX commands inside included). T37c fix 1: the closing
#: `$` may be missing ("Pal$^{4}*"), and one letter glued straight after
#: the group belongs to it ("Vignolini $^{13}C"): a name never continues
#: with a letter right after an affiliation mark.
_SCRIPT_GROUP = re.compile(r"\$[\^_]\{(?:[^{}]|\{[^{}]*\})*\}\$?(?:[A-Za-z](?![A-Za-z]))?")
#: `$^a$`, `$^*$`: a group written without braces.
_BARE_SCRIPT = re.compile(r"\$[\^_][^$\s]{0,12}\$")
#: An ORCID icon read as the letters "ID" glued to a surname ("MiuraID1,2").
_GLUED_ORCID = re.compile(r"(?<=[a-zà-ÿ])ID(?![a-z])")


# --- helpers (after the main function, per CLAUDE.md) -----------------------

def align(raw: str, characters: Sequence) -> Tuple[Dict[int, int], bool]:
    """
    Pair each printed character of the text with its entry in `characters`.

    Whitespace is skipped on the text side (the list has none). The PDF's
    line-end hyphen code `\\x02` is a "-" in the list. A non-printing
    character the list skipped is fine; a printing one it lacks is not.

    Returns:
        ({text index: character index}, True only if every printed character
        on both sides was paired).
    """
    mapping: Dict[int, int] = {}
    index = 0
    unpaired = 0
    for position, letter in enumerate(raw):
        if letter.isspace():
            continue
        if index < len(characters) and (
                characters[index].text == letter
                or (letter == "\x02" and characters[index].text == "-")):
            mapping[position] = index
            index += 1
        elif letter.isprintable():
            unpaired += 1
    return mapping, unpaired == 0 and index == len(characters)


def script_kinds(characters: Sequence) -> List[Optional[str]]:
    """
    For each character: "^" (superscript), "_" (subscript) or None.

    Each character is compared with the last NON-script character before it
    (the letter the script hangs on), so a run "a,b,1" is judged letter by
    letter against the name it follows.
    """
    kinds: List[Optional[str]] = []
    anchor = None
    for character in characters:
        kind = _kind_against(character, anchor) if anchor is not None else None
        kinds.append(kind)
        if kind is None:
            anchor = character
    return kinds


def _kind_against(character, anchor) -> Optional[str]:
    """Is `character` a script (or a prime) hanging on `anchor`? Which way?"""
    anchor_height = anchor.rect.y1 - anchor.rect.y0
    height = character.rect.y1 - character.rect.y0
    if anchor_height <= 0:
        return None
    shift = character.baseline - anchor.baseline       # top-left: raised < 0
    if not (BASELINE_SHIFT * anchor_height < abs(shift) < NEXT_LINE_SHIFT * anchor_height):
        return None
    if character.rect.x0 < anchor.rect.x1 - TUCK_BACK * anchor_height:
        return None                                    # not to its right: another line
    if _is_prime_coded_as_zero(character, anchor, shift, height, anchor_height):
        return PRIME
    if height >= SCRIPT_HEIGHT * anchor_height:
        return None
    return SUPERSCRIPT if shift < 0 else SUBSCRIPT


def _is_prime_coded_as_zero(character, anchor, shift: float, height: float,
                            anchor_height: float) -> bool:
    """
    T37c fix 3: a maths font that draws the prime (′) with the code of "0".

    PMC9295807's "2′-deoxyguanosine" is stored as "2" + "0": the "0" is in
    another font than the "2", raised by 0.38 of its height and NOT smaller
    (a real superscript zero is set smaller, and is caught as a script). It
    must also be glued to the character before it (no space).
    """
    return (character.text == "0" and not character.space_before
            and shift < 0 and height >= SCRIPT_HEIGHT * anchor_height
            and bool(character.font) and character.font != anchor.font)


def write_marked(raw: str, mapping: Dict[int, int],
                 kinds: Sequence[Optional[str]]) -> Tuple[str, int]:
    """
    The text with each run of same-kind script characters wrapped.

    A run ends at whitespace or a change of kind, so "a,b" is one group and
    "a, b" two.
    """
    out: List[str] = []
    open_kind: Optional[str] = None
    runs = 0
    for position, letter in enumerate(raw):
        kind = kinds[mapping[position]] if position in mapping else None
        if kind == PRIME:
            # A prime is written as itself, never grouped (T37c fix 3).
            kind, letter = None, "′"
        if kind != open_kind:
            if open_kind is not None:
                out.append("}$")
            if kind is not None:
                out.append(f"${kind}{{")
                runs += 1
            open_kind = kind
        out.append(letter)
    if open_kind is not None:
        out.append("}$")
    return "".join(out), runs
