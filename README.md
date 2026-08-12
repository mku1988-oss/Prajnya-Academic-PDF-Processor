# Shastrasaara V6

Offline academic-PDF processing for literature review: text, tables and figures
extracted into a structured, RAG-ready form — running entirely on a single 8GB GPU.

**Status: working end to end.** Measured on a 9-document, 485-page corpus:
**48.9 minutes, 0 failures**, with **58.6% of all pages never reaching the GPU**.
The 8-page benchmark runs in **160s against V5's 410s** and passes the parity
gate at 36/38 (see [Performance](#performance)).

---

## Contents

- [Quick start](#quick-start) · [The CLI](#the-cli) · [Recipes](#recipes)
- [Books](#books-cost-hours-so-they-are-asked-about) · [Metadata review](#metadata-review)
- [Output](#what-you-get) · [Performance](#performance) · [Testing](#testing)

---

## Quick start

```bash
# 0. Activate the project's environment (every command below assumes it)
conda activate shastrasaara

# 1. Start the model server (leave it running)
python run_llama_server.py

# 2a. Process a paper from the command line
python -u run_prajnya.py --pdf "paper.pdf"

# 2b. ...or open the graphical interface instead
python -u run_prajnya_studio.py
```

Step 0 is not optional, and step 1 is not `llama-server.exe` typed by hand —
both for reasons worth knowing before something fails oddly. See
[The Python environment](#the-python-environment) and
[Starting the model server](#starting-the-model-server).

---

## Two front ends, one pipeline

| entrypoint | what it is |
|---|---|
| `run_prajnya.py` | the command line — one PDF, a folder, or the review screen |
| `run_prajnya_studio.py` | Prajnya Studio, the graphical interface (opens in your browser) |

Both end up in `src/processing/runner.py`; the GUI's call arrives there through
a subprocess worker so a long batch cannot freeze the interface. Neither front
end reimplements "process a document".

```bash
run_prajnya_studio.py                 # default http://127.0.0.1:8850
run_prajnya_studio.py --port 8973     # a different port
run_prajnya_studio.py --no-browser    # don't open a tab
```

**The default port is 8850, not 8080, on purpose** — 8080 is the local
llama-server's, so defaulting there would stop the GUI starting on exactly the
machine that is set up correctly to run the pipeline. If the port you pick is
already busy the launcher refuses and says so; it does not start and quietly
serve nothing.

### Prajnya Studio — the eight screens

| screen | shortcut | what it is for |
|---|---|---|
| **Library / Batch** | `Ctrl+L` | Type a folder path, **Scan** to classify every PDF on the CPU (~1s each, no GPU), then **Start**. Live transcript, hard **Stop**, and the book question asked as a dialog. |
| **Results** | | Metadata, page text, the figure gallery grouped by printed label, tables beside their reconstructed grids, and the raw JSON. **V5 had no results screen at all** — 107 figures and 31 tables were invisible. |
| **Ledger** | `Ctrl+G` | Every document ever processed, searchable. Per-row edit, CSV and BibTeX export. |
| **Review Queue** | | Whatever is flagged incomplete, with exactly which fields are missing. Same correction logic as the CLI's `--review`, so the two cannot disagree. |
| **Chapter Map** | | Books only: every chapter's authors and start page, and a per-chapter **Read** button. |
| **Strategy Lab** | `Ctrl+K` | Which model does which job. Edits the real config files, shows a diff before every save, backs up what it touches, and warns before you assign two models your card cannot hold together. |
| **Settings** | `Ctrl+,` (quick panel) | Default paths, book policy, recursion, PDF backend, and the light/dark theme. A blank value never overrides a default. |
| **Help** | | Quick start and FAQ. |

Scanning a folder works with no model server running; **processing pages does
not** — start `llama-server` first, exactly as for the CLI.

A run is **one subprocess per PDF**, so a batch cannot freeze the interface and
**Stop** is a real kill of the document currently running, not a flag checked
between documents. Only one run is active at a time (there is one GPU); opening
a second tab joins the run in progress rather than starting another.

---

## The CLI

`run_prajnya.py` is the command-line surface. One PDF, a folder of them, or the
review screen.

```bash
run_prajnya.py --pdf <file|folder> [options]
run_prajnya.py --review
```

### What to process

| flag | what it does |
|---|---|
| `--pdf PATH` | a PDF, **or a folder** of PDFs. Required unless `--review`. |
| `--recursive` | with a folder, search subdirectories too |
| `--pages "2-5"` | only these 1-based pages. Accepts `4`, `2-5`, `1 3-5`, `2-5,7` |
| `--force` | re-process even if the ledger already has this document |
| `--review` | review flagged metadata and stop. **No PDF is read, no model is called.** |

### Books (see [Books](#books-cost-hours-so-they-are-asked-about))

| flag | what it does |
|---|---|
| *(omitted)* | front matter and chapter map only — seconds, no GPU |
| `--chapters 3 4` | read those chapters. Accepts `3-5` ranges too |
| `--chapters` *(no numbers)* | read **every** chapter — hours of GPU on a full volume |
| `--max-pages N` | hard ceiling on body pages read, whatever else was asked |

### Everything else

| flag | default | what it does |
|---|---|---|
| `--out PATH` | `run_outputs/` inside this repo | where results go — a JSON path for one PDF, a run directory for a folder |
| `--noninteractive` | off | never prompt. Books take the cheap default; review is deferred to a later `--review` |
| `--front-matter-pages N` | 10 | leading pages scanned for DOI / ISBN / ISSN |
| `--backend NAME` | `pypdfium2` | PDF backend (`pypdfium2` or `pymupdf`) |
| `--verbose` | off | DEBUG logging |

### Exit codes

| code | meaning |
|---|---|
| `0` | everything processed |
| `1` | nothing could run — no such file, no ledger to review |
| `2` | finished, but at least one document failed. **A batch never stops for one bad PDF.** |

---

## Recipes

```bash
# One paper, everything default
run_prajnya.py --pdf "paper.pdf"

# Just the pages you care about
run_prajnya.py --pdf "paper.pdf" --pages 1 4-7

# A whole folder, left running unattended overnight
run_prajnya.py --pdf "C:/papers" --recursive --noninteractive

# Re-do a folder from scratch, ignoring what the ledger already knows
run_prajnya.py --pdf "C:/papers" --force --out "C:/papers/rerun"

# A book: what is in it, for free
run_prajnya.py --pdf "volume.pdf"

# A book: read chapters 3 to 5
run_prajnya.py --pdf "volume.pdf" --chapters 3-5

# A book: read all of it, but never more than 40 pages
run_prajnya.py --pdf "volume.pdf" --chapters --max-pages 40

# Fix the metadata that came out incomplete, whenever you have time
run_prajnya.py --review
```

---

## Books cost hours, so they are asked about

The difference between a paper and a 345-page volume is minutes versus hours of
GPU. So V6 never guesses:

- **A folder is pre-flighted first.** Every PDF is classified on the CPU (~1s
  each) and the whole list is printed — books included — **before any GPU work
  starts.** A 345-page volume hiding in a batch is visible while it still costs
  nothing to change your mind.
- **Then the batch runs without interruption**, so it can be left alone.
- **The default is the cheap one.** Omitting `--chapters` gives the front matter
  and the chapter map: title, editors, publisher, identifiers, and every chapter
  with its own authors and start page. Seconds, no GPU.
- **`--noninteractive` always resolves cheap.** Nobody is watching to stop an
  accidental two-hour read.

Per-chapter authorship is the point: in an edited volume every chapter has
different authors, so a passage retrieved from chapter 3 must cite *its* authors,
not the editor. On the sample volume, 335 of 345 pages carry their own chapter's
authors.

### The policy file

`config/processing_policy.yaml` holds the same setting for the CLI and the GUI,
so neither front end needs to know the other exists.

```yaml
books:
  policy: ask         # skip | ask | all | chapters
  chapters: []        # used only when policy is `chapters`
  max_pages: null     # ceiling on body pages, whatever the policy
  implausible_chapter_pages: 80
```

`ask` is a *policy value*, and asking is a **capability the caller supplies**:
the CLI supplies a terminal prompt, a GUI supplies a dialog, a cron job supplies
nothing and falls back to `skip`. Precedence is CLI flag → per-run override →
this file → the built-in default, and **every fallback resolves to the cheap
option**.

---

## Metadata review

A citation you cannot resolve is not a citation. Every processed document is
assessed, written to the ledger, and flagged if something required is missing.

**What is required depends on what the document is:**

| type | must have |
|---|---|
| `research_article` | title + authors + DOI + journal |
| `book` | title + (editors \| authors) + (publisher \| ISBN) + year |
| `book_chapter` | title + authors + (journal \| publisher) + DOI |

The either-or groups are bibliography, not laxity: an edited volume is cited by
its editors and a monograph by its authors. A book needs no DOI — many have none
and the ISBN is the identifier that matters.

A record is also flagged when validation confidence is below 0.75, or when the
merge marked the authors suspicious.

**When you are asked:**

| situation | what happens |
|---|---|
| one document, at a terminal | you are asked straight away |
| a batch | flags accumulate; you are asked **once, at the end** |
| `--noninteractive` | nothing asks; the log says how many are waiting |
| any time later | `run_prajnya.py --review` picks up everything still flagged |

Corrections go into the ledger *and* back into the affected output JSON — and
only the metadata block. Pages, figures and tables are never touched by a
correction.

### The ledger

`persistence/semantic_ledger.csv` (override with `SARA_LEDGER_PATH`) is the
permanent record: one row per document, keyed by the file's **SHA-256**, so
renaming a PDF does not create a duplicate and editing one is correctly treated
as new. A document already in the ledger is skipped unless `--force`.

---

## What you get

```
run_outputs/                      ← always here, whatever your working directory
├── documents/<name>.v6.json      one per document
├── batch_summary.json            stable name, always the latest
├── batch_summary_<UTC>.json      timestamped, kept
├── figures/<document>/           p3_fig1.png … + manifest.json
└── tables/<document>/            recovered table crops
```

The location is anchored to this repository, not to your PDFs and not to the
directory you launched from — so results are always in one findable place, and
a reference library never accumulates generated JSON. `--out` overrides it.

Each document JSON:

```jsonc
{
  "document_info":  { "filename", "document_type", "processing_time", "success" },
  "metadata_final": { "title", "authors", "doi", "journal", "year", "editors", … },
  "review_status":  "reviewed | provisional",
  "pages": [ { "page_number", "content", "quality_score",
               "metadata": { "processing_method", "coverage", "figures", "tables",
                             "chapter": { "number", "title", "authors" } } } ],
  "summary": { "errors", "page_count", "figure_count" },
  "profile": { … the un-flattened record, so nothing is lost … }
}
```

`processing_method` tells you what each page cost: `pymupdf` (text layer, free),
`glm_fullpage` (a VLM read), `glm_repair` (a table crop re-read),
`two_shot_title_page`.

Figure crops carry a `manifest.json` naming each one and its caption, which is
what links a cropped figure back to the text that discusses it.

---

## Performance

Full sample corpus, measured on a **verified clean GPU** (0 MiB resident before
the run — see [the VRAM rule](#the-vram-rule)):

| document | type | pages | time | s/page |
|---|---|---:|---:|---:|
| NMR benchmark | article | 8 | 160.3s | 20.0 |
| Collagen structure | article | 25 | 488.5s | 19.5 |
| Collagen review | article | 11 | 168.3s | 15.3 |
| CollagenFibrilAssembly | book_chapter | 36 | 366.0s | 10.2 |
| Hydration-induced | article | 20 | 199.0s | 9.9 |
| JRuberti | article | 26 | 174.7s | 6.7 |
| Saturated Solutions | article | 7 | 234.1s | 33.4 |
| Silberbush-2019 | article | 7 | 117.9s | 16.8 |
| **Tendon assembly** | **book** | **345** | **1026.7s** | **3.0** |
| **total** | | **485** | **48.9 min** | **6.1** |

**How the 485 pages were read:** 58.6% from the text layer, 34.6% a full VLM
read, 3.1% a table repair, 1.6% the title two-shot. 88 figures and 17 tables
recovered, zero errors.

Two numbers are worth dwelling on:

- **A book is cheaper per page than a paper** — 3.0s against 14.8s — because 246
  of its 345 pages were read on the CPU. That is the entire V6 thesis, measured.
- **The slowest document is the one with no text layer.** *Saturated Solutions*
  is an old scan: 0 pages harvestable, 4 of 7 needing table repair, 33.4s/page.
  The pre-pass cannot help a scan, and does not pretend to.

---

## Why V6 exists

V5 adopted better OCR models and got *slower*: 413s on the 8-page benchmark
against the older pipeline's 324s. The reason is structural — on one 8GB GPU
wall time is just total GPU compute, and V5 added a second model (layout
grounding) plus crop re-reads on top of the reading pass. Concurrency was tested
four ways and every arrangement netted zero or worse.

**V6's thesis: the only lever is not making the VLM call at all.**

```
        ┌──────────── CPU PRE-PASS (all pages, ~1s total) ─────────────┐
 PDF ──►│ DocumentCache │ text harvest │ PageClassifier │ LayoutDetector│
        └──────────────┬────────────────────────────────────────────────┘
                       ▼
      PagePlan per page: {reader: text | vlm_crops | vlm_full,
                          table_boxes, figure_boxes, scanned?}
```

The layout detector drops from a 4.9GB VLM at ~17s/page to an ONNX model at
~600ms/page on the CPU. The GPU is invoked only where vision is genuinely
required.

---

## Design principles

**VLM-agnostic.** No model name, model-specific prompt or vendor quirk appears
in `src/`. Everything model-specific lives in a profile YAML, resolved through a
role registry (`extraction`, `layout_detector`, `title_detection`). Swapping
models is a config change.

**Pluggable seams.** `LayoutDetector` returns `List[Detection]` whether an ONNX
model or a grounding VLM produced them. `PdfBackend` abstracts PDF text and
geometry. **Asking is a capability the caller supplies** — `book_prompt.py` and
`review_prompt.py` hold conversations and no decisions, so a GUI replaces those
two files and nothing else.

**Permissive licences only.** Apache-2.0 / MIT / BSD. AGPL and "community
licence" components are banned — this ruled out DocLayout-YOLO (AGPL-3.0 via
Ultralytics) and HunyuanOCR-1.5 (Tencent Community Licence), both otherwise
attractive.

**Every stage is gated.** `tests/test_parity_nmr.py` asserts metadata, table
recovery, figure crops and word coverage against archived known-good output. No
optimization ships without passing it.

---

## Setup

```bash
python -m pip install -r requirements.txt
```

Model weights are not tracked in git — see [`models/README.md`](models/README.md).
Zotero credentials go in `.env` (`ZOTERO_API_KEY`, `ZOTERO_LIBRARY_ID`); without
them the run degrades loudly to no-Zotero rather than silently.

### The Python environment

Every command runs inside the `shastrasaara` conda environment (Python 3.10),
so **activate it first and keep it activated**:

```bash
conda activate shastrasaara
python run_llama_server.py --check
```

**Why this matters more than it looks.** Without the environment active, PATH
may fall through to another Python that has all the same packages at *different
major versions* (numpy 2.0 vs 1.26 — measured, on the development machine's
miniconda base). It does not fail with `ImportError`; it runs and quietly
produces different numbers. On a project whose gates are IoU thresholds and
coverage ratios, that is the worst possible failure mode.

Both entrypoints therefore **check the interpreter before importing anything**
and stop with an explanation rather than a traceback. If `conda activate
shastrasaara` cannot find the environment by name, it is registered by path
instead — `conda env list` prints where it is, and naming its interpreter
directly always works:

```bash
<env>/python.exe run_llama_server.py --check
```

### Starting the model server

Use `run_llama_server.py` rather than invoking `llama-server` yourself:

```bash
python run_llama_server.py
```

It pins the server's working directory to this repository and checks that every
model file named in `config/vlm_profiles/model.ini` actually exists before
starting. Both matter because llama.cpp resolves the preset's model paths
against **the server process's own working directory**, and `--models-preset`
validates nothing at startup — so a server started from the wrong folder
launches happily, passes the pipeline's pre-flight, and then fails in the middle
of a run at the first page that needs a model.

The `llama-server` binary is not part of this repository. The launcher finds it
via `--llama-server <path>`, the `SARA_LLAMA_SERVER` environment variable (put
it in `.env`), or `PATH`.

### The VRAM rule

**Never compare two timings taken at different VRAM states.** The 8GB card
cannot hold the title model (~5.5GB) and the body model (~2.5GB) together, and
the router sheds models by count rather than by memory. A model left resident by
the previous document makes the next one spill to system RAM and decode ~3×
slower, with byte-identical output — invisible in the results, visible only on
the clock. The same benchmark measured 148s clean and 260s contaminated.

Launch with `--models-max 1`, and record
`nvidia-smi --query-gpu=memory.used --format=csv,noheader` alongside any timing
you intend to quote.

### Confirming a run stayed offline

Grep the log for `openrouter` (must be 0 hits) and `localhost:8080` (must be
> 0).

---

## Testing

**The test suite is not part of a release.** It lives in the development
repository, where it is run before anything ships:

```bash
python -m pytest tests/ -q          # 1316 tests, no GPU needed

# The parity gate against a fresh run
python -m pytest tests/test_parity_nmr.py -v \
    --run-output "run_outputs/documents/<benchmark>.v6.json"
```

It is kept out of the release deliberately rather than by oversight. A large
part of it cannot run outside the development machine at all: the parity gate
needs the benchmark PDF and a live llama-server, several tests read the sample
corpus, and the deployment tests import `tools/`, which a release also does not
contain. Shipping a suite that fails on a clean clone says less about the
project's quality than not shipping one.

---

## Relationship to V5

V5 lives at `../v4/gemini_refactor` and keeps running. It is a **separate git
repository** — V6 work does not go there. The decomposition table in
[`plan.md`](plan.md) is the migration order; roughly 900 lines of V5's strategy
file were unreachable dead code and were never carried over.

---

## Licence

[Apache-2.0](LICENSE). Third-party attributions in [`NOTICE`](NOTICE).

**PyMuPDF is AGPL-3.0 and is therefore not the shipping path.** Apache-2.0 code
can be pulled into an AGPL work, but distributing that combination would
relicense the whole product as AGPL. So **pypdfium2 is the default backend**;
PyMuPDF remains an optional local-development backend, never appears in
`requirements.txt`, and is never required for a run to succeed. All PDF access
goes through `PdfBackend` — `fitz` is never imported directly.
