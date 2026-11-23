"""
The journal name as read from a title page: cleaned, or flagged as doubtful.

For biology researchers: at citation level `local` nobody vouches for the
journal; it is whatever the title page printed where a journal name usually
stands. Two misreadings are common (ledger T34, `GATE_S0B.md`):

  - the running header, with the volume and date glued on:
    "TRENDS in Genetics Vol.20 No.1 January 2004". The journal is the part
    before "Vol.", so that tail is cut off;
  - a section word that the publisher prints above the title: "REVIEWS",
    "Research Article". That is not a journal at all and cannot be repaired
    from the page, so it is kept but flagged for a person to look at.

An abbreviation ("Biochem. J.") is what the page prints and is left alone.
Only names from a fixed list of section words are flagged: journals print
their own names in capitals too ("GENETICS", "CELL").
"""
from __future__ import annotations

import re
from typing import Any, Tuple

# Words publishers print above an article's title to say what kind of piece it
# is. A journal name that is EXACTLY one of these (any case) is a misreading.
SECTION_WORDS = frozenset({
    "review", "reviews", "review article", "mini review", "minireview", "mini-review",
    "article", "articles", "research article", "research articles", "original article",
    "original research", "original research article", "research", "research paper",
    "report", "reports", "research report", "brief report", "letter", "letters",
    "communication", "communications", "short communication", "brief communication",
    "rapid communication", "perspective", "perspectives", "commentary", "editorial",
    "highlights", "highlight", "technical note", "methods", "method", "opinion",
    "essay", "primer", "news", "news and views", "feature", "features", "focus",
    "spotlight", "correspondence", "analysis", "resource", "tools and resources",
    "full paper", "full papers", "paper", "papers", "invited review", "special issue",
})

_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
# A running header's tail: "Vol.20 No.1 January 2004", "Volume 12, Issue 3, 2019".
_HEADER_TAIL = re.compile(
    r"[\s,;]+(?:vol(?:ume)?\.?\s*\d+)"
    r"(?:[\s,;]*(?:no\.?|issue|iss\.?)\s*\d+)?"
    r"(?:[\s,;]*" + _MONTH + r")?"
    r"(?:[\s,;]*(?:1[89]|20)\d{2})?"
    r"[\s,;.]*$", re.I)


def check_journal_name(name: Any) -> Tuple[Any, bool]:
    """
    A title-page journal name, cleaned, and whether it is doubtful.

    Args:
        name: the journal name as read (any type; non-strings come back as is).

    Returns:
        `(name, suspect)`: the name with a running header's volume / issue /
        date tail removed, and True when what remains is a section word.

    Example:
        >>> check_journal_name("TRENDS in Genetics Vol.20 No.1 January 2004")
        ('TRENDS in Genetics', False)
        >>> check_journal_name("REVIEWS")
        ('REVIEWS', True)
        >>> check_journal_name("Biochem. J.")
        ('Biochem. J.', False)
    """
    if not isinstance(name, str) or not name.strip():
        return name, False
    cleaned = strip_header_tail(name.strip())
    return cleaned, is_section_word(cleaned)


def strip_header_tail(name: str) -> str:
    """The name without a trailing "Vol. n [No. n] [month] [year]"; unchanged if nothing is left."""
    cut = _HEADER_TAIL.sub("", name).strip(" ,;")
    return cut if cut else name


def is_section_word(name: str) -> bool:
    """Whether the whole name is a section word ("REVIEWS", "Research Article")."""
    return re.sub(r"\s+", " ", name.strip(" .:;,")).lower() in SECTION_WORDS
