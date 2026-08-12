#!/usr/bin/env python3
"""
The number a figure is called by in the paper, as opposed to where it sits.

For biology researchers: every figure we save gets an id like `p3_fig1` —
page 3, first figure found there. That id is stable and unique, which is what
a filename needs, but it is **not what the paper calls the figure**. On the
NMR benchmark the figure on page 3 is printed as "Figure 2"; on page 4 it is
"Figure 3". Measured across the sample corpus, **45 of 107 figures have an id
whose number disagrees with the printed one.**

That gap matters for retrieval. When the body text says "as shown in Fig. 2",
a RAG has to reach the image — and nothing in the record connects the phrase
"Fig. 2" to `p3_fig1`. This module supplies that link by reading the label off
the front of the caption we already captured from the text layer.

WHY ONLY THE CAPTION. The obvious extension is to scan the page's text layer
for a "Figure N" line near the figure. Measured, that recovers a further 9
figures and agrees with the caption **15/15 where both exist** — but it also
walks straight into the trap this codebase has already met once, in
`figure_escalation`: **a figure is routinely CITED pages away from where it is
printed.** In the tendon volume, page 20 reads "…packing is affected
(Fig. 2.2). The rigid COL2 domain…" — a citation in running prose, on a page
whose own figure is 2.2's neighbour, not 2.2. A nearest-line rule attaches
that citation as the label and is confidently wrong, which is worse than
having no label at all. A caption box, by contrast, was already paired to its
figure geometrically before we got here.

So: the caption is trusted, the page is not. The text-layer fallback needs a
caption-versus-citation discriminator before it can be safe, and that is not
built yet.
"""

import re
from typing import Optional, Tuple

#: What a figure label looks like at the start of its caption.
#:
#: The number may be plain (`3`), chapter-scoped (`2.1`, in edited volumes
#: where each chapter restarts numbering), or roman (`IV`). The trailing
#: boundary matters: without it, "Figs" would match "Figure" and `1` would
#: match the `1` of `1.2`.
_LABEL = re.compile(
    r"^\s*"
    r"(?P<kind>Fig(?:ure|s|ures)?|FIG|Scheme|Chart|Plate|Panel)"
    r"\s*\.?\s*"
    r"(?P<number>[0-9]{1,3}(?:\.[0-9]{1,3})*|[IVXLivxl]{1,6})"
    r"(?![0-9A-Za-z])",
    re.IGNORECASE,
)

#: Every spelling of "figure" collapses to one word for matching. A paper
#: writes "Fig. 2" in the body and "Figure 2" in the caption — or the reverse —
#: and a RAG joining the two must not care which.
_KIND_ALIASES = {
    "fig": "figure", "figs": "figure", "figure": "figure", "figures": "figure",
    "scheme": "scheme", "chart": "chart", "plate": "plate", "panel": "panel",
}


def parse_printed_label(caption: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Read the printed label from the front of a figure's caption.

    For biology researchers: given "Fig. 2 | Mechanisms of hydration…" this
    returns the label as printed ("Fig. 2") and a normalized key
    ("figure 2"). The key is what makes retrieval work — the caption may say
    "Fig. 2" while the body says "Figure 2", and both must match.

    Only the START of the caption is examined. A number mentioned later in the
    caption ("…compared with Figure 5") describes a different figure.

    Args:
        caption: the caption text, or None/empty when none was captured.

    Returns:
        `(printed_label, label_key)` — for example `("Fig. 2.1", "figure 2.1")`.
        `(None, None)` when the caption is missing or carries no label, which
        is the honest answer for the 57 of 107 figures whose captions came
        back empty or as a panel fragment.
    """
    if not caption:
        return None, None

    match = _LABEL.match(caption)
    if not match:
        return None, None

    kind = match.group("kind")
    number = match.group("number")

    # The printed form is the matched text itself, so "Fig." keeps its full
    # stop and "Figure" does not acquire one. Rebuilding it from the groups
    # would silently drop the period, since the separator sits outside them.
    printed = _tidy(match.group(0))
    key = f"{_KIND_ALIASES.get(kind.lower().rstrip('.'), 'figure')} {number.lower()}"

    return printed, key


def _tidy(printed: str) -> str:
    """Collapse the whitespace a two-column text layer sprinkles into a label."""
    return " ".join(printed.split())
