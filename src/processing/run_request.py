#!/usr/bin/env python3
"""
What one run was asked to do — as data, with no terminal attached.

For biology researchers: this is the "order form" for a run. Which PDFs, where
the results go, which pages, whether to re-do work already done. It is plain
data with no behaviour, which is exactly the point.

**Why an order form rather than passing arguments around.** The CLI builds one
of these from its command line; the GUI will build one from its screen; and a
worker process will rebuild one from a JSON file. All three then hand it to the
same runner. If instead each front end passed its own arguments down, "process
a document" would end up implemented more than once, and the two copies would
start disagreeing — which is the single failure this whole seam exists to
prevent.

**It must survive being written to a file and read back.** A run will later
happen in a separate process (one per PDF, so a hung document can be killed),
and the request travels as JSON. Hence `to_dict` / `from_dict`, and hence
`Path` values are written as plain strings.

⚠️ **The `--chapters` sentinel.** `chapters=None` means "read no chapter
bodies" (the safe, cheap default); `chapters=[]` means "read EVERY chapter",
which for a 345-page volume is hours of GPU. Those two are *not*
interchangeable, and `[]` is falsy — so `to_dict` writes the key
unconditionally and no code here may test it for truthiness. This exact
sentinel collision once made a 345-page book "finish" in 1.5 seconds having
read nothing at all, with exit code 0 (see `router._book_body_pipeline`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .extractors.front_matter import DEFAULT_FRONT_MATTER_PAGES
from .paths import (
    CONFIG_ROOT,
    DEFAULT_OUTPUT_ROOT,
    resolve_batch_output,
    resolve_document_output,
)


@dataclass
class RunRequest:
    """
    Everything a run needs to know, and nothing about how it was asked for.

    Attributes:
        paths: the PDFs to process, already resolved (a folder has been
            expanded into its files by the caller).
        config_root: the `config/` directory the strategies read.
        out_root: the run directory — `documents/`, `figures/`, `tables/` and
            the batch summary all live under it.
        out_document: an explicit output file for a SINGLE document, when the
            user named one. None means "derive it from `out_root`".
        backend: PDF backend name, or None for the default (pypdfium2).
        pages: 1-based page numbers to process, or None for all.
        chapters: BOOKS ONLY. None reads no bodies; an empty-but-present list
            reads every chapter. See the module docstring — this is a sentinel.
        max_pages: BOOKS ONLY. Ceiling on body pages read.
        front_matter_pages: how many leading pages to scan for DOI/ISBN/ISSN.
        force: re-process even if the ledger already has the document, and
            re-derive metadata rather than returning the stored row.
        recursive: with a folder, search subdirectories too.
        noninteractive: never prompt; books take the cheap default.
        verbose: DEBUG logging, and tracebacks on a failed document.
        ledger_path: an explicit ledger file, or None for the default.
        book_overrides: a per-run policy override — how a GUI passes a choice
            without writing the config file. Same keys as the YAML `books` block.
    """

    paths: List[Path] = field(default_factory=list)
    config_root: Path = CONFIG_ROOT
    out_root: Path = DEFAULT_OUTPUT_ROOT
    out_document: Optional[Path] = None
    backend: Optional[str] = None
    pages: Optional[List[int]] = None
    chapters: Optional[List[int]] = None
    max_pages: Optional[int] = None
    front_matter_pages: int = DEFAULT_FRONT_MATTER_PAGES
    force: bool = False
    recursive: bool = False
    noninteractive: bool = False
    verbose: bool = False
    ledger_path: Optional[Path] = None
    book_overrides: Optional[Dict[str, Any]] = None

    @classmethod
    def from_args(cls, args: Any, paths: Optional[List[Path]] = None) -> "RunRequest":
        """
        Build a request from the command line.

        This is the ONE place that decides where output lands, for both shapes
        of run — a single PDF and a folder of them. Keeping that decision in a
        single function is what stops the two entry points drifting apart, which
        is how a run's results go missing.

        Args:
            args: the parsed command line (an `argparse.Namespace`, or anything
                carrying the same attribute names). `args.pages` and
                `args.chapters` must already be parsed into integer lists.
            paths: the PDFs, when the caller has already expanded a folder.
                None means `args.pdf` is a single file.

        Returns:
            A `RunRequest`. For a folder, `out_root` is the run directory and
            each document's JSON is derived beneath it; for a single PDF with
            an explicit `--out`, `out_document` is that exact path.
        """
        target = Path(args.pdf)
        is_batch = paths is not None

        if is_batch:
            out_root = resolve_batch_output(args.out)
            out_document = None
        else:
            paths = [target]
            # An explicit --out for one PDF names the JSON file itself. Its
            # FOLDER becomes the run root, so the figures and tables of that
            # document land beside it rather than back in the default tree —
            # one run, one place. Without --out, everything goes to the
            # repository's run_outputs/, as it always has.
            out_document = (resolve_document_output(target, args.out)
                            if args.out else None)
            out_root = (out_document.parent if out_document is not None
                        else DEFAULT_OUTPUT_ROOT)

        return cls(
            paths=list(paths),
            config_root=CONFIG_ROOT,
            out_root=out_root,
            out_document=out_document,
            backend=args.backend,
            pages=args.pages,
            chapters=args.chapters,
            max_pages=args.max_pages,
            front_matter_pages=args.front_matter_pages,
            force=args.force,
            recursive=args.recursive,
            noninteractive=args.noninteractive,
            verbose=args.verbose,
        )

    def output_for(self, pdf_path: Path) -> Path:
        """
        Where this PDF's JSON is written.

        An explicit single-document path wins; otherwise the file is named
        after the PDF and placed in the run's `documents/` directory.
        """
        if self.out_document is not None:
            return self.out_document
        return self.out_root / "documents" / f"{pdf_path.stem}.v6.json"

    def to_dict(self) -> Dict[str, Any]:
        """
        The request as plain JSON-safe data, for handing to another process.

        Every key is written unconditionally. `chapters` in particular must
        never be written behind an `if` — see the module docstring; `[]` and
        `None` mean opposite things and one of them is falsy.
        """
        return {
            "paths": [str(path) for path in self.paths],
            "config_root": str(self.config_root),
            "out_root": str(self.out_root),
            "out_document": str(self.out_document) if self.out_document else None,
            "backend": self.backend,
            "pages": list(self.pages) if self.pages is not None else None,
            "chapters": list(self.chapters) if self.chapters is not None else None,
            "max_pages": self.max_pages,
            "front_matter_pages": self.front_matter_pages,
            "force": self.force,
            "recursive": self.recursive,
            "noninteractive": self.noninteractive,
            "verbose": self.verbose,
            "ledger_path": str(self.ledger_path) if self.ledger_path else None,
            "book_overrides": self.book_overrides,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RunRequest":
        """Rebuild a request written by `to_dict` — the other half of the trip."""
        chapters = raw.get("chapters")
        pages = raw.get("pages")
        return cls(
            paths=[Path(value) for value in raw.get("paths") or []],
            config_root=Path(raw.get("config_root") or CONFIG_ROOT),
            out_root=Path(raw.get("out_root") or DEFAULT_OUTPUT_ROOT),
            out_document=_optional_path(raw.get("out_document")),
            backend=raw.get("backend"),
            pages=list(pages) if pages is not None else None,
            chapters=list(chapters) if chapters is not None else None,
            max_pages=raw.get("max_pages"),
            front_matter_pages=raw.get("front_matter_pages",
                                       DEFAULT_FRONT_MATTER_PAGES),
            force=bool(raw.get("force")),
            recursive=bool(raw.get("recursive")),
            noninteractive=bool(raw.get("noninteractive")),
            verbose=bool(raw.get("verbose")),
            ledger_path=_optional_path(raw.get("ledger_path")),
            book_overrides=raw.get("book_overrides"),
        )


@dataclass
class DocumentOutcome:
    """
    What happened to one document — the child's report to the parent.

    Deliberately plain data: no `DocumentProfile`, no pydantic models, nothing
    that would have to survive a trip between processes. The ledger is written
    from this, in the parent, by `runner.record_outcome`.

    Attributes:
        path: the source PDF.
        output: the JSON that was written, or None if nothing was.
        elapsed: wall time in seconds.
        metadata: the merged bibliographic record, as a plain dict.
        document_type: what the document was classified as.
        errors: non-fatal errors the pipeline recorded and continued past.
        error: the fatal error that stopped this document, or None.
        skipped: why the document was skipped, or None.
    """

    path: Path
    output: Optional[Path] = None
    elapsed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_type: Any = None
    errors: List[str] = field(default_factory=list)
    error: Optional[str] = None
    skipped: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """True when the document was processed and written."""
        return self.error is None and self.skipped is None


# --- helpers (after the main classes, per CLAUDE.md) ------------------------

def _optional_path(value: Any) -> Optional[Path]:
    """A path, or None — without turning None into `Path('.')`."""
    return Path(value) if value else None
