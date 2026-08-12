"""The book strategy — front matter today, chapter reading later."""

from .chapter_boundaries import ChapterMap, ChapterSpan, detect_chapter_spans
from .front_matter_record import BookRecord, build_book_record
from .strategy import BookStrategy
from .toc import TocEntry, parse_toc_entries, toc_entries_to_text

__all__ = [
    "BookRecord",
    "ChapterMap",
    "ChapterSpan",
    "BookStrategy",
    "TocEntry",
    "build_book_record",
    "detect_chapter_spans",
    "parse_toc_entries",
    "toc_entries_to_text",
]
