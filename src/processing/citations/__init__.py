"""
Citations: how a document is cited, and how much leaves the machine to find out.

For biology researchers: one run setting, `citations.level`, chooses between
three ways of establishing a paper's bibliographic record:

    local    Prajnya's own reading of the PDF. Nothing leaves the machine.
    zotero   checked against the researcher's Zotero library.
    online   Crossref / OpenAlex, plus the paper's reference list.

Plan and reasoning: `docs/citation_levels_plan.md`. No model and no GPU is
used anywhere in this package.
"""

from .block import build_citations_block, metadata_source_of, source_label
from .level import (CITATION_LEVELS, DEFAULT_LEVEL, LEVEL_LOCAL, LEVEL_ONLINE,
                    LEVEL_ZOTERO, resolve_level)

__all__ = [
    "CITATION_LEVELS",
    "DEFAULT_LEVEL",
    "LEVEL_LOCAL",
    "LEVEL_ONLINE",
    "LEVEL_ZOTERO",
    "build_citations_block",
    "metadata_source_of",
    "resolve_level",
    "source_label",
]
