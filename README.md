# Shastrasaara V6

Offline academic-PDF processing for literature review: text, tables and figures
extracted into a structured, RAG-ready form — running entirely on a single 8GB GPU.

**Status: working end to end, and measured.** On a 9-document, 485-page corpus
it produced **0 failures, 107 figures and 31 tables**, with **61.9% of all pages
never reaching the GPU** at all. The same corpus has been run twice — once from
the command line and once through the graphical interface — with **identical
output, document for document, and zero metadata differences**.

Everything here runs **offline**. No page, no title and no citation is sent
anywhere; the only network call the pipeline makes is an optional Zotero lookup
you configure yourself.

---

## Contents

- [What's new in V6](#whats-new-in-v6)
- [Requirements](#requirements) · [Installation](#installation) · [Quick start](#quick-start)
- [Two front ends](#two-front-ends-one-pipeline) · [Prajnya Studio](#prajnya-studio--the-eight-screens) · [The CLI](#the-cli) · [Recipes](#recipes)
- [Books](#books-cost-hours-so-they-are-asked-about) · [Metadata review](#metadata-review) · [What you get](#what-you-get)
- [Performance](#performance) · [How it works](#how-it-works) · [Design principles](#design-principles)
- [Configuration](#configuration) · [Troubleshooting](#troubleshooting) · [Running notes](#running-notes)
- [Testing](#testing) · [Relationship to v5.x](#relationship-to-v5x) · [Licence](#licence)

---

## What's new in V6

V6 is a rewrite, not an increment. The previous release (v5.x) had adopted
better OCR models and become **slower** — because on a single 8GB GPU, wall
time is simply total GPU compute, and each better model added compute rather
than removing it. V6 starts from the opposite premise: **the only real lever is
not making the model call at all.**

### The engine

| | v5.x | **V6** |
|---|---|---|
| 8-page benchmark | 413s | **194s** |
| how pages are read | every page to the VLM | **61.9% never reach the GPU** |
| layout detection | a second VLM held in VRAM | **a 131 MB CPU ONNX model** |
| a 345-page book | not supported in practice | **345 pages in 23.5 min**, 4.1s/page |

A cheap CPU pre-pass reads the text layer, classifies each page and detects
layout, then decides the **minimum** set of VLM calls. Vision is used only where
vision is genuinely required — which is why a book, mostly plain prose, is
*cheaper per page* than a dense paper.

### New capabilities

- **Prajnya Studio, rebuilt** — eight screens including a **results viewer**,
  which v5.x did not have at all: 107 figures and 31 tables were extracted and
  then invisible. Also new: a review queue, a book chapter map, and a **dark
  theme** (v5.x offered the menu item but it changed nothing).
- **Books are first-class.** Front matter, a parsed contents page, per-chapter
  authorship, and chapter-scoped reading. A passage from page 178 of an edited
  volume cites *that chapter's* authors, not the editors'.
- **Documents are classified before anything is processed**, and routed to a
  research-article, book, or book-chapter pipeline. A whole folder is
  pre-flighted on the CPU first, so a 345-page volume hiding in a batch is
  visible while it still costs nothing.
- **A permanent ledger**, keyed by file SHA-256, with a metadata review flow.
  Renaming a PDF does not create a duplicate.
- **Strategy Lab** — change which model does which job, from the interface,
  with a diff before every save and a **Restore Defaults** that always works.
- **Figures are grouped by their printed label**, so the three crops of
  "Figure 4" arrive as one figure rather than three.

### Correctness work you would not otherwise see

- **Every figure on a page containing equations used to be silently discarded**
  — a classification rule returned before the visual scan ran. Fixing it
  recovered **+22% more figures** across the corpus (88 → 107). Nothing had
  failed; `figure_count: 0` simply looked like an honest answer.
- **A reference list draws a box around every "CrossRef" link**, and clustered
  those look exactly like a figure. Measured word-density separates them
  cleanly (decoration 51–72% coverage, real figures 0–8%), so bibliography
  pages no longer go to the GPU.
- **`--force` now genuinely re-derives metadata** instead of re-reading every
  page and then returning the value the first run had cached.

### Licensing

V6 ships only **OSI-permissive** components. PyMuPDF (AGPL-3.0) is no longer on
the shipping path, and DocLayout-YOLO (AGPL-3.0, via Ultralytics) was dropped
in favour of PP-DocLayoutV3 (Apache-2.0). See [Licence](#licence).

---

## Requirements

| | |
|---|---|
| **OS** | Windows 10/11 or Linux. Developed and measured on Windows 11. |
| **Python** | **3.10** — pinned. Both entrypoints refuse to start on anything else, for a reason worth reading in [The Python environment](#the-python-environment). |
| **GPU** | An **8 GB** NVIDIA card is enough, and is what every number here was measured on. More VRAM lets you run larger models; see [Configuration](#configuration). |
| **[llama.cpp](https://github.com/ggml-org/llama.cpp)** | `llama-server`, built with your GPU backend. Not bundled — it is a separate project with its own build. |
| **Disk** | **~13 GB** of model weights (measured: 12.7 GB of GGUF pairs + a 131 MB layout model), plus your output. |

A CPU-only machine will run the pre-flight scan, the ledger, and every screen of
the interface, but cannot process pages.

---

## Installation

**1. Get the code and the Python environment.**

```bash
git clone https://github.com/mku1988-oss/Prajnya-Academic-PDF-Processor.git
cd Prajnya-Academic-PDF-Processor

conda create -n shastrasaara python=3.10
conda activate shastrasaara
pip install -r requirements.txt
```

**2. Download the models.** Weights are not in the repository — they are tens of
gigabytes. [`models/README.md`](models/README.md) lists every file, its licence,
and where to put it. The short version: one `curl` for the 131 MB layout model,
and three `model`/`mmproj` GGUF pairs (6 files, 12.7 GB) into `models/<vendor>/`.

| model | role | licence |
|---|---|---|
| PP-DocLayoutV3 (ONNX) | layout detection, on the CPU | Apache-2.0 |
| GLM-OCR | page reading and table repair | MIT / Apache-2.0 |
| olmOCR-2-7B | title pages, scanned fallback | Apache-2.0 |
| DeepSeek-OCR-2 | figure second opinion | Apache-2.0 |

**3. Point the launcher at `llama-server`**, once:

```bash
# in .env, or your environment
SARA_LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server
```

**4. Check the install before running anything real:**

```bash
python run_llama_server.py --check
```

It resolves every model path in `config/vlm_profiles/model.ini` and names
anything missing. A clean result means the configuration and the files agree.

**Optional — Zotero enrichment.** Put `ZOTERO_API_KEY` and `ZOTERO_LIBRARY_ID`
in `.env` to let metadata be cross-checked against your library. Without them
the run says so plainly and continues; nothing degrades silently.

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
| NMR benchmark | article | 8 | 194.7s | 24.3 |
| Collagen structure | article | 25 | 456.5s | 18.3 |
| Collagen review | article | 11 | 186.3s | 16.9 |
| CollagenFibrilAssembly | book_chapter | 36 | 491.1s | 13.6 |
| Hydration-induced | article | 20 | 321.9s | 16.1 |
| JRuberti | article | 26 | 306.4s | 11.8 |
| Saturated Solutions | article | 7 | 279.5s | 39.9 |
| Silberbush-2019 | article | 7 | 139.5s | 19.9 |
| **Tendon assembly** | **book** | **345** | **1410.3s** | **4.1** |
| **total** | | **485** | **63.7 min** | **7.8** |

**How the 485 pages were read:**

| route | pages | share |
|---|---:|---:|
| text layer — **no GPU at all** | 300 | **61.9%** |
| full VLM read | 162 | 33.4% |
| table repair (a crop re-read) | 15 | 3.1% |
| title page, two-shot | 8 | 1.6% |

**107 figures and 31 tables recovered, zero errors, and 0 of 9 documents flagged
for review.** Every crop referenced by a manifest was present on disk: 138
referenced, 138 found, 0 orphaned.

Three numbers are worth dwelling on:

- **A book is cheaper per page than a paper** — 4.1s against ~17s — because most
  of its 345 pages were read on the CPU. That is the entire V6 thesis, measured.
- **The slowest document is the one with no text layer.** *Saturated Solutions*
  is an old scan: nothing harvestable, most pages needing table repair,
  39.9s/page. The pre-pass cannot help a scan, and does not pretend to.
- **The same corpus through the graphical interface produced identical output** —
  same 485 pages, same 107 figures, same 31 tables, and **zero metadata
  differences** across title, DOI, journal, year, authors, publisher and
  validation fields for all nine documents. Two front ends, one pipeline, and
  the claim is measured rather than asserted.

> **Timings are only comparable on a clean GPU.** The same benchmark measures
> 148s or 260s depending purely on whether a model was left resident by the
> previous run — with byte-identical output. See [the VRAM rule](#the-vram-rule).

---

## How it works

The previous release adopted better OCR models and got *slower*: 413s on the 8-page benchmark
against its own predecessor's 324s. The reason is structural — on one 8GB GPU
wall time is just total GPU compute, and it had added a second model (layout
grounding) plus crop re-reads on top of the reading pass. Concurrency was tested
in four configurations and every one netted zero or worse: scheduling
redistributes compute, it never removes it.

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

**Every stage is gated.** A parity harness asserts metadata, table recovery,
figure crops and word coverage against archived known-good output. No
optimization ships without passing it — see [Testing](#testing).

---

## Configuration

Everything model-specific lives in `config/`, and **no model name appears
anywhere in `src/`.** The pipeline asks only for a *role*; the configuration
decides which model answers. Swapping a model is therefore a configuration
change, not a code change — which is what lets the tool outlive the models it
was written against.

### The six roles

| role | what it does | shipped default |
|---|---|---|
| `default` | general reading | `olmocr-2-7b` |
| `title_detection` | the title-page two-shot | `olmocr-2-7b` |
| `extraction` | body pages and table repair | `glm-ocr` |
| `layout_detector` | finds figures and tables | `onnx` (CPU) |
| `figure_escalation` | second opinion where the cheap pass contradicts itself | `deepseek-ocr-v2` |
| `understanding` | reserved; unused today | `olmocr-2-7b` |

Deleting a role's line disables that feature rather than breaking the run.

### The three coupled layers

| file | holds |
|---|---|
| `config/research_article_strategy.yaml` | the role registry, and the pipeline's own settings |
| `config/vlm_profiles/<model>.yaml` | prompt mode, fixed prompts, target DPI, VRAM estimate, co-residency |
| `config/vlm_profiles/model.ini` | the llama-server preset: GGUF paths, context size, sampling |

They must agree — a profile naming a preset that does not exist is a run that
fails at the first page. **Strategy Lab checks all three**, shows a diff before
every save, backs up the file it touches, and warns before you assign two models
your card cannot hold at once. It can also duplicate a profile, so you can point
the tool at a model this codebase has never heard of without editing Python.

`config/factory_defaults/` is a frozen copy of the shipped configuration. Every
tab's **Restore Defaults** reads from it, so no sequence of edits is
unrecoverable. Nothing ever writes *into* that directory.

### Environment variables

| variable | effect |
|---|---|
| `SARA_LLAMA_SERVER` | where the `llama-server` binary is |
| `SARA_LEDGER_PATH` | keep the ledger outside the repository |
| `SARA_CACHE_PATH` | move the Zotero cache |
| `SARA_GUI_STORAGE_SECRET` | makes GUI sessions survive a restart |
| `ZOTERO_API_KEY`, `ZOTERO_LIBRARY_ID` | enable Zotero enrichment |

All are read from `.env` if present. An unset variable resolves to nothing and
says so — it never becomes the literal string `${VAR}`.

---

## Troubleshooting

Every entry here is a failure that has actually happened, not a hypothetical.

| symptom | cause and fix |
|---|---|
| **"the wrong Python is running this file"** | Something other than the project environment started it. `conda activate shastrasaara`, then re-run. The message prints which interpreter it actually got. |
| **A run dies mid-way with a model error, after pre-flight passed** | `llama-server` was started from the wrong directory, so its preset's relative model paths resolve to nothing — and it validates none of them at startup. Start it with `python run_llama_server.py`. |
| **Everything is ~3× slower than the table above, with identical output** | A model from the previous run is still resident and the current one has spilled to system RAM. Start the server with `--models-max 1` (the launcher's default) and see [the VRAM rule](#the-vram-rule). |
| **"Port 8850 is already in use"** | Prajnya Studio is probably open in another window. Otherwise `--port 8851`. |
| **The interface loads but processing fails immediately** | `llama-server` is not running. Scanning a folder needs no server; reading pages does. |
| **A document is re-processed although it ran before** | The ledger is keyed by file SHA-256, so an edited PDF is correctly a new document. To force it either way, use `--force`. |
| **Zotero metadata is not being used** | `ZOTERO_API_KEY` / `ZOTERO_LIBRARY_ID` are unset. The log says so explicitly at startup rather than degrading quietly. |
| **A book "finished" in seconds** | That is the default: front matter and the chapter map only. Add `--chapters` to read bodies — and see what that costs first. |

---

## Running notes

Four things that are worth understanding before something behaves oddly.

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

## Relationship to v5.x

**v5.x is still on `main` and still runs.** V6 is a rewrite rather than an
upgrade: the output schema, the configuration layout and the GUI are all
different, and there is no in-place migration. Run whichever you need — they
share no state except, if you point them at it, a Zotero library.

If you are coming from v5.x, the differences that will bite first:

| | v5.x | V6 |
|---|---|---|
| CLI entrypoint | `run_Prajnya.py` | `run_prajnya.py` |
| GUI entrypoint | `prajnya_studio/studio_app.py` | `run_prajnya_studio.py` |
| GUI port | 8080 | **8850** (8080 is llama-server's) |
| configuration | inside `prajnya_studio/` | one `config/` tree |
| output | beside the PDF, or beside your shell | always `run_outputs/` in the repo |
| ledger | none written | `persistence/semantic_ledger.csv` |

Roughly 900 lines of v5.x's strategy code were unreachable and were not carried
over.

---

## Licence

[Apache-2.0](LICENSE). Third-party attributions in [`NOTICE`](NOTICE).

**PyMuPDF is AGPL-3.0 and is therefore not the shipping path.** Apache-2.0 code
can be pulled into an AGPL work, but distributing that combination would
relicense the whole product as AGPL. So **pypdfium2 is the default backend**;
PyMuPDF remains an optional local-development backend, never appears in
`requirements.txt`, and is never required for a run to succeed. All PDF access
goes through `PdfBackend` — `fitz` is never imported directly.
