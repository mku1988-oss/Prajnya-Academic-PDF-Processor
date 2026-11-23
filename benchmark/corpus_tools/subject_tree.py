"""
Springer Nature's subject taxonomy, to place a paper in a field by its keywords.

For biology researchers: Nature, Nature Communications and Scientific Reports
print no field label in their JATS XML, but their keywords are terms of the
publisher's own subject taxonomy ("Cartilage development", "Chromatin
structure"), a tree whose branches include "Biochemistry", "Biophysics",
"Developmental biology", "Computational biology and bioinformatics",
"Physics" and "Chemistry". Following a paper's keywords up that tree places
it by the publisher's classification, not ours.

Source: SciGraph subjects taxonomy, 2017-02-15 (CC BY 4.0), from
github.com/springernature/scigraph, `2017Q1/models/taxonomies/subjects/`.
Fetched into `corpus/cache/taxonomy/` (not committed; re-fetchable).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

TAXONOMY_URL = ("https://raw.githubusercontent.com/springernature/scigraph/master/2017Q1/models/"
                "taxonomies/subjects/springernature-scigraph-subjects.2017-02-15.ttl")
TAXONOMY_FILE = Path(__file__).resolve().parent.parent / "corpus" / "cache" / "taxonomy" / \
    "sn-subjects-2017-02-15.ttl"

#: The branches a paper can be placed in: our field name -> the taxonomy's id.
TARGETS = {
    "biochemistry": "biochemistry",
    "biophysics": "biophysics",
    "developmental biology": "developmental-biology",
    "computational biology": "computational-biology-and-bioinformatics",
    "physics": "physics",
    "chemistry": "chemistry",
}

_BLOCK = re.compile(r"^subjects:(\S+)\n(.*?)(?=^\S|\Z)", re.S | re.M)
_LABELS = re.compile(r'skos:(?:prefLabel|altLabel)\s+(.+?)\s*[;.]\s*$', re.M)
_BROADER = re.compile(r"skos:broader\s+(.+?)\s*[;.]\s*$", re.M)


class SubjectTree:
    """Term labels and their broader terms, read from the taxonomy file."""

    def __init__(self, path: Path = TAXONOMY_FILE) -> None:
        self.broader: Dict[str, List[str]] = {}
        self.by_label: Dict[str, Set[str]] = {}
        text = Path(path).read_text(encoding="utf-8")
        for match in _BLOCK.finditer(text):
            term, body = match.group(1), match.group(2)
            for line in _BROADER.findall(body):
                self.broader.setdefault(term, []).extend(re.findall(r"subjects:([\w\-]+)", line))
            for line in _LABELS.findall(body):
                for label in re.findall(r'"([^"]+)"@en', line):
                    self.by_label.setdefault(fold(label), set()).add(term)

    def ancestors(self, term: str) -> Set[str]:
        """The term and every broader term above it."""
        seen, stack = set(), [term]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.broader.get(current, []))
        return seen

    def fields_of(self, keywords: Iterable[str]) -> Dict[str, List[str]]:
        """{field: [the keywords that lead to it]} over the TARGETS branches."""
        found: Dict[str, List[str]] = {}
        for keyword in keywords:
            for term in self.by_label.get(fold(keyword), ()):
                above = self.ancestors(term)
                for field, target in TARGETS.items():
                    if target in above and keyword not in found.get(field, []):
                        found.setdefault(field, []).append(keyword)
        return found

    def field_of(self, keywords: Iterable[str]) -> str | None:
        """The ONE field a paper's keywords reach (A11 rule), or None.

        Per keyword, a biology sub-field beats its parent: a keyword reaching
        biochemistry does not also count for chemistry, one reaching
        biophysics not for physics. A paper whose keywords reach two fields,
        or none, gets None (not eligible).
        """
        reached: Set[str] = set()
        for keyword in keywords:
            fields = {field for field, words in self.fields_of([keyword]).items() if words}
            if "biochemistry" in fields:
                fields.discard("chemistry")
            if "biophysics" in fields:
                fields.discard("physics")
            reached |= fields
        return next(iter(reached)) if len(reached) == 1 else None

    def known(self, keyword: str) -> bool:
        return fold(keyword) in self.by_label


def fold(label: str) -> str:
    """Case and dash folding, so "Stem-cell differentiation" meets "stem cell differentiation"."""
    return re.sub(r"[\s\-‐]+", " ", label).strip().lower()
