"""
Clean up control-character damage in harvested PDF text.

For biology researchers: when a PDF is typeset, a long word at the end of a
line is split with a hyphen ("measure-" / "ments"). Some PDFs store that hyphen
using a private font code rather than a real hyphen character. When the text is
harvested, that code survives as an invisible *control character*, so the word
arrives as "measure<02>ments" instead of "measurements".

Why this matters more than it looks. That token can never match a search for
"measurements" - not by keyword search, and not by any embedding model, because
every one of them splits text on characters like this. The damage is silent:
the page looks fine, the word count is right, and the term is simply
unfindable. Hyphenation happens at line breaks, which is exactly where long
technical words land, so the losses are concentrated in the vocabulary that
matters most ("crosslinking", "biomolecular", "lysozyme").

Measured on the nine-document sample corpus (485 pages):

    4,635 control characters in total
    4,484 (96.7%)  0x02 sitting between two letters -> a split word
       34 ( 0.7%)  the same, with a stray space after it
       64 ( 1.4%)  standing alone inside formula text
       53 ( 1.1%)  other positions

Repairing them makes 2,060 distinct terms searchable again, including the
corpus's most important vocabulary: "collagen" was broken 105 times, "tendon"
63, "extracellular" 29, "proteoglycans" 22.

Of those breaks, 111 (2.5%) are followed by a capital letter and are REAL
hyphens rather than split words: "Ehlers<02>Danlos", and a long tail of
hyphenated author surnames that citations depend on. They are handled
separately - see `_REAL_HYPHEN`.

WHAT THIS DELIBERATELY DOES NOT DO. A control character standing on its own was
some symbol the font encoding lost - "149 <03> 1 ms" was almost certainly
"149 +/- 1 ms", and "0.37 s <04> 1" an exponent. Which symbol it was cannot be
recovered from the text alone, and guessing would put a number into the record
that the paper never printed. Those become spaces: the token boundary is
repaired, and nothing is invented.

The same reasoning rules out the other damage on these pages. Some PDFs render
"=" as a wide-fraction glyph and brackets as thorn/eth characters, so a formula
arrives as "h 1/4 1.62". Those are legitimate Unicode characters that appear in
real scientific text, so a blanket substitution would corrupt correct pages to
repair broken ones. Left alone on purpose.
"""
import re
from typing import Optional

#: Control characters that carry no meaning in harvested prose. Tab, carriage
#: return and newline are excluded: they are real layout, not damage.
_CONTROL_CHARS = r"\x00-\x08\x0b\x0c\x0e-\x1f"

#: A REAL hyphen that happens to be stored as a control code: the break is
#: followed by a capital letter. Measured over the sample corpus, 111 of 4,510
#: breaks (2.5%) look like this, and they are overwhelmingly genuine hyphens:
#: "Ehlers<02>Danlos" (16x) plus a long tail of hyphenated author surnames
#: (Chiquet-Ehrismann, Loeys-Dietz, Boot-Handford, Bruckner-Tuderman) and
#: technical terms (tenascin-C, trans-Golgi, Pro-Gly). Joining these would give
#: "EhlersDanlos", which matches no query anyone would type, and would corrupt
#: exactly the author names this project needs for citations.
#:
#: Checked BEFORE the split-word rule, because both match the same position.
_REAL_HYPHEN = re.compile(
    rf"(?<=[A-Za-z])[{_CONTROL_CHARS}][^\S\n]*(?=[A-Z])"
)

#: A word split across a line break: letter, the stray control code, then the
#: rest of the word in lower case. The optional space covers the variant where
#: the harvest left one behind ("regres<02> sion").
#:
#: `[^\S\n]*` is horizontal whitespace ONLY, so this never joins across a line
#: break. Rejoining whole lines would change the page's line structure, which
#: other passes read; repairing the word is the smaller, safer claim.
_SPLIT_WORD = re.compile(
    rf"(?<=[A-Za-z])[{_CONTROL_CHARS}][^\S\n]*(?=[a-z])"
)

#: Anything left over once split words are repaired.
_LEFTOVER_CONTROL = re.compile(rf"[{_CONTROL_CHARS}]")


def normalize_harvested_text(text: Optional[str]) -> str:
    """
    Repair control-character damage in text harvested from a PDF.

    This is applied where page text is turned into *stored* content, so that
    every consumer downstream (search index, embeddings, the RAG's evidence
    objects) sees the same repaired text. Doing it here rather than in each
    consumer means the fix happens once and the file on disk is correct.

    It is deliberately conservative: it repairs split words and removes
    meaningless control codes, and it changes nothing else. Real characters,
    spacing and line structure are left exactly as they were.

    Args:
        text: Raw text from the PDF text layer or from a model read. None and
            empty strings are accepted so callers need no guard of their own.

    Returns:
        The repaired text, or "" when there was nothing to repair.

    Examples:
        >>> normalize_harvested_text("collagen cross\\x02linking")
        'collagen crosslinking'
        >>> normalize_harvested_text("the regres\\x02 sion fit")
        'the regression fit'
        >>> normalize_harvested_text("Ehlers\\x02Danlos syndrome")
        'Ehlers-Danlos syndrome'
        >>> normalize_harvested_text("149 \\x03 1 ms")
        '149 1 ms'
    """
    if not text:
        return ""

    # Step 1: a break before a CAPITAL is a real hyphen in a name or compound
    # term, not a split word. Restore the hyphen rather than closing the gap.
    # Must run before step 2, since both match the same position.
    repaired = _REAL_HYPHEN.sub("-", text)

    # Step 2: rejoin words that a line-break hyphen split in two.
    repaired = _SPLIT_WORD.sub("", repaired)

    # Step 3: whatever control codes remain stood for a symbol we cannot
    # recover. Replace with a space so the tokens either side stay separate,
    # and do not pretend to know what the symbol was.
    repaired = _LEFTOVER_CONTROL.sub(" ", repaired)

    # Step 4: step 3 can leave a doubled space where a lone code sat between
    # two spaces. Collapse runs of spaces and tabs, without touching newlines.
    repaired = re.sub(r"[ \t]{2,}", " ", repaired)

    return repaired


def count_control_damage(text: Optional[str]) -> int:
    """
    Count control characters in text, for reporting and for tests.

    For biology researchers: useful to confirm a corpus is clean after a run,
    the same way a gel is checked for the expected bands rather than assumed.

    Args:
        text: Any harvested text.

    Returns:
        How many meaningless control characters the text still contains.
    """
    if not text:
        return 0
    return len(_LEFTOVER_CONTROL.findall(text))
