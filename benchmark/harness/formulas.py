"""
Formula scoring (amendment A11 §6): display and inline formulas.

For biology researchers: the answer key gives each formula as the
publisher's MathML. A tool usually writes LaTeX (`\\frac{a}{b}`). Both are
reduced to one plain token sequence (identifiers, numbers, operators, in
reading order) and compared: 1.0 means the same tokens in the same order.
Layout-only markup (spacing, styles, grouping rows) is not a token, so a
formula typed differently but reading the same scores 1.0.

LaTeX goes to MathML first with latex2mathml (MIT), so both sides are read
by the same tokeniser.
"""
from __future__ import annotations

import re
import unicodedata

from lxml import etree
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

CONTEXT_WORDS = 8              # prose words before a formula that place it
CONTEXT_MATCH = 0.6            # least context agreement for a key/tool pair
# LaTeX command names that look like words in a tool's context ("\mathrm").
LATEX_WORDS = {"mathrm", "mathbf", "mathit", "text", "frac", "left", "right", "quad",
               "mathcal", "operatorname", "math", "display", "block", "begin", "end",
               "cdot", "times", "alpha", "beta", "gamma", "delta", "lambda", "sigma",
               "mu", "partial", "nabla", "sum", "int", "sqrt", "tag", "hat", "tilde"}

# MathML leaves that carry content: identifier, number, operator, text.
TOKEN_TAGS = {"mi", "mn", "mo", "mtext", "ms"}
# Structures whose reading order is not their child order on the page;
# a marker token keeps "a over b" distinct from "a b".
STRUCTURE_TOKENS = {"mfrac": "/", "msqrt": "sqrt", "mroot": "root",
                    "msup": "^", "msub": "_", "msubsup": "_^",
                    "munder": "_", "mover": "^", "munderover": "_^"}
# Characters that are the same symbol written two ways.
SAME_SYMBOL = str.maketrans({"\u2212": "-", "\u2013": "-", "\u00d7": "*", "\u22c5": "*",
                             "\u00b7": "*", "\u2032": "'", "\u2009": "", "\u00a0": "",
                             "\u2061": "", "\u2062": "", "\u2063": ""})
INVISIBLE = {"", "\u2061", "\u2062", "\u2063"}


def formula_similarity(key_mathml: str, tool_formula: str | None) -> float | None:
    """Token-sequence similarity in [0, 1] between the key and a tool's formula.

    `tool_formula` is LaTeX or MathML (MathML when it starts with '<').
    None when the tool gave nothing for this formula (scored as 0 by the
    caller, never averaged as missing); 0.0 when it cannot be read.
    """
    if tool_formula is None:
        return None
    key_tokens = mathml_tokens(key_mathml)
    tool_tokens = (mathml_tokens(tool_formula) if tool_formula.lstrip().startswith("<")
                   else latex_tokens(tool_formula))
    if not key_tokens and not tool_tokens:
        return 1.0
    return Levenshtein.normalized_similarity(key_tokens, tool_tokens)


def score_formulas(key_formulas: list[dict], tool_formulas: dict) -> dict:
    """Per kind ('display', 'inline'): mean similarity over the key's formulas.

    `tool_formulas` maps a key formula id to the tool's formula text (the
    matching by position is done before this call). A formula the tool did
    not give scores 0. A kind with no key formula is None (not applicable).
    """
    result = {}
    for kind in ("display", "inline"):
        scores = [formula_similarity(f["mathml"], tool_formulas.get(f["id"])) or 0.0
                  for f in key_formulas if f["kind"] == kind]
        result[f"formula_{kind}"] = sum(scores) / len(scores) if scores else None
        result[f"formula_{kind}_count"] = len(scores)
    return result


def match_formulas(key_formulas: list[dict], tool_formulas: list[dict] | None) -> dict:
    """Pair the tool's formulas with the key's by position: key formula id -> tool text.

    Both lists are in reading order and carry `context_before`, the text
    just before each formula. The pairing keeps reading order (no
    crossings) and maximises the summed context agreement; a pair needs at
    least CONTEXT_MATCH agreement. Only the words of the context are
    compared (inline math in it is written differently by every tool).
    The formula's own content plays no part, so a badly read formula is
    still found, and scored low, not dropped.
    """
    if not key_formulas or not tool_formulas:
        return {}
    keys = [context_words(f.get("context_before")) for f in key_formulas]
    tools = [context_words(f.get("context_before")) for f in tool_formulas]
    best = [[0.0] * (len(tools) + 1) for _ in range(len(keys) + 1)]
    for i in range(1, len(keys) + 1):
        for j in range(1, len(tools) + 1):
            agreement = context_agreement(keys[i - 1], tools[j - 1])
            paired = best[i - 1][j - 1] + agreement if agreement >= CONTEXT_MATCH else -1.0
            best[i][j] = max(best[i - 1][j], best[i][j - 1], paired)
    return trace_pairs(best, keys, tools, key_formulas, tool_formulas)


# ---------------------------------------------------------------------------
# Helpers (after the main functions, per the project rule)
# ---------------------------------------------------------------------------

def mathml_tokens(mathml: str) -> list[str]:
    """The content tokens of a MathML fragment, in reading order."""
    try:
        root = etree.fromstring(strip_entities(mathml).encode("utf-8"),
                                etree.XMLParser(recover=True, resolve_entities=False))
    except etree.XMLSyntaxError:
        return []
    if root is None:
        return []
    tokens: list[str] = []
    walk(root, tokens)
    return join_numbers(tokens)


def join_numbers(tokens: list[str]) -> list[str]:
    """One number, one token: publishers often mark 0.0192 up as three
    pieces (<mn>0</mn><mo>.</mo><mn>0192</mn>) where a LaTeX converter
    gives one <mn>. Digit runs and a decimal point between digits are joined."""
    joined: list[str] = []
    for token in tokens:
        previous = joined[-1] if joined else ""
        number_so_far = re.fullmatch(r"\d+(\.\d*)?", previous) is not None
        if number_so_far and token.isdigit():                 # "0." + "0192", "12" + "3"
            joined[-1] = previous + token
        elif token == "." and previous.isdigit():            # "0" + "."
            joined[-1] = previous + token
        else:
            joined.append(token)
    # A point that no digit followed was an operator after all ("x = 3.").
    tidy: list[str] = []
    for token in joined:
        tidy.extend([token[:-1], "."] if token.endswith(".") and token[:-1].isdigit() else [token])
    return tidy


def walk(element, tokens: list[str]) -> None:
    """Depth-first: a leaf gives its text tokens; a structure adds its marker
    between its first child and the rest (a/b -> a / b, x^2 -> x ^ 2)."""
    tag = local_name(element)
    if tag in ("annotation", "annotation-xml"):        # TeX source kept beside the MathML
        return
    if tag in TOKEN_TAGS:
        tokens.extend(leaf_tokens(element.xpath("string()"), tag))
        return
    children = [c for c in element if isinstance(c.tag, str)]
    marker = STRUCTURE_TOKENS.get(tag)
    for position, child in enumerate(children):
        if marker and position == 1:
            tokens.append(marker)
        walk(child, tokens)


def leaf_tokens(text: str, tag: str) -> list[str]:
    """A number stays whole ("3.14"); a multi-letter identifier ("sin", "exp")
    stays whole; an operator or text run is split into its symbols/words."""
    # NFKC folds styled letters (bold v, U+1D42F) to plain ones: the key marks
    # style with an attribute, LaTeX converters with a different code point.
    text = unicodedata.normalize("NFKC", text).translate(SAME_SYMBOL).strip()
    if text in INVISIBLE:
        return []
    if tag == "mn":
        return [text]
    # Letters one by one: publishers write a subscript "DD" as two <mi>, a
    # LaTeX "\text{DD}" gives one <mtext>; "sin" is s, i, n on both sides.
    return re.findall(r"\d+(?:\.\d+)?|\S", text)


def latex_tokens(latex: str) -> list[str]:
    """LaTeX -> MathML (latex2mathml) -> tokens; [] when it cannot be converted."""
    from latex2mathml.converter import convert
    source = latex.strip().strip("$").strip()
    source = re.sub(r"^\\\[|\\\]$|^\\\(|\\\)$", "", source).strip()
    source = re.sub(r"\\(?:begin|end)\{(?:equation|align|gather|displaymath)\*?\}", "", source)
    # An equation number is a label, not formula content (the key keeps it
    # in <label>): "\tag{2}", "\qquad (3)", a trailing "(4a)".
    source = re.sub(r"\\tag\*?\{[^}]*\}", "", source)
    source = re.sub(r"(?:\\q?quad|\\hfill|~|\s)*\(\s*\d+[a-z]?\s*\)\s*$", "", source).strip()
    try:
        return mathml_tokens(convert(source))
    except Exception:                     # latex2mathml raises many kinds on bad input
        return []


def trace_pairs(best, keys, tools, key_formulas, tool_formulas) -> dict:
    """Walk the alignment table back from the end, collecting the pairs."""
    pairs, i, j = {}, len(keys), len(tools)
    while i > 0 and j > 0:
        agreement = context_agreement(keys[i - 1], tools[j - 1])
        if agreement >= CONTEXT_MATCH and best[i][j] == best[i - 1][j - 1] + agreement:
            pairs[key_formulas[i - 1]["id"]] = tool_formulas[j - 1].get("text")
            i, j = i - 1, j - 1
        elif best[i][j] == best[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return pairs


def context_words(text: str | None) -> str:
    """The last CONTEXT_WORDS prose words (3+ letters) of a context, lower case."""
    words = re.findall(r"[^\W\d_]{3,}", unicodedata.normalize("NFKC", text or "").lower())
    words = [w for w in words if w not in LATEX_WORDS]
    return " ".join(words[-CONTEXT_WORDS:])


def context_agreement(key_words: str, tool_words: str) -> float:
    """0..1 agreement of two contexts' words (0 when either has none)."""
    if not key_words or not tool_words:
        return 0.0
    return fuzz.ratio(key_words, tool_words) / 100.0


def strip_entities(text: str) -> str:
    """Named entities (&InvisibleTimes;) are unknown to a plain XML parser."""
    return re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)\w+;", "", text)


def local_name(element) -> str:
    tag = element.tag if isinstance(element.tag, str) else ""
    return tag.split("}", 1)[-1].split(":")[-1]
