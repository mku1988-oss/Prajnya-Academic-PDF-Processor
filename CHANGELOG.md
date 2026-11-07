# Changelog

What changed in each release of Prajnya. Versions are tagged in this
repository (`git tag`), and every output JSON records the version that produced
it in `document_info.pipeline_version`.

---

## 6.2.1

### Output

- **Every body page records how long it took to read**, as `page_time` (seconds)
  in its metadata. The title page already recorded its `total_time`.

### Documentation

- **The README's performance table now reports cost per kind of page** (title
  page, text layer, model read, table page, scanned page) instead of seconds per
  page for each document. Per document it misled: an 8-page paper looked slower
  per page than a 25-page one, because every document pays a fixed start-up
  cost that weighs more on a short one.

---

## 6.2.0

### Tables

- **A table printed as two sub-tables is read in full.** Some papers print two
  blocks under one heading (for example "At 20 °C" and "At 25 °C"), close enough
  that they are detected as one table. The reading model used to stop after the
  first block, losing everything below it; each block is now read and the rows
  combined. On the sample corpus this recovered 64 missing values in one table.
- **Very long tables are no longer cut off.** A 55-row table was stored ending at
  row 41, because the page image and the transcription share the reading
  model's working memory and the transcription ran out of room. Table reads now
  have the space they need.

### Prajnya Studio

- **The "Ledger File" setting now takes effect.** It was saved but never used:
  a batch, the Ledger screen and the Review Queue all read the default ledger
  whatever the setting said.
- **"Restore Defaults" restores a working configuration.** The stored defaults
  had fallen behind the live configuration, so restoring them could leave a
  model-server setting that current `llama-server` builds reject, and undo the
  long-table fix above.

### Documentation

- The README is reorganised and shorter; release history moved here.

---

## 6.1.0

### Figures

- Captions printed on the page after their figure are found.
- Figures printed side by side in two columns are no longer merged into one.
- 26 captions get back their opening sentence, which is the part a search
  matches on; panel letters are no longer stored as captions.
- **97% of figure crops now carry their printed label** (was 89%).

### Tables

- **Every table now leaves with its reconstructed grid.** Only tables that went
  through the repair path used to be transcribed, so 14 of the sample corpus's
  31 tables had no grid and their numbers reached no search index.

### New page fields, for retrieval layers

- `headings`: the typography of each line that looks like a heading (size,
  weight and capitalisation relative to the body text).
- `reference_density`, `has_reference_heading`: how much a page reads like a
  bibliography.
- `page_type`: recorded on every page.

### Other

- DeepSeek-OCR-2 loads again on current `llama-server` builds.
- The release version is now checked across the repository, so it cannot fall
  out of step again. (Some of these changes were first published on
  2026-11-04 without a version bump; 6.1.0 is the version that includes them.)

---

## 6.0.0

A rewrite, not an increment. The previous release (5.x) adopted better OCR
models and became **slower**, because on a single 8 GB GPU processing time is
essentially total model work, and each better model added work rather than
removing it. 6.0.0 starts from the opposite premise: **the most effective
speed-up is not making the model call at all.**

| | 5.x | **6.0.0** |
|---|---|---|
| 8-page benchmark | 413s | **194s** |
| how pages are read | every page by the vision model | **61.9% never reach the GPU** |
| layout detection | a second vision model held in memory | **a 131 MB CPU model** |
| a 345-page book | not supported in practice | **23.5 min, 4.1 s/page** |

### New

- **Figures and tables as retrievable objects**: cropped, captioned, given a
  stable identity, panels grouped under their printed label, tables with a
  reconstructed grid.
- **Prajnya Studio, rebuilt**: eight screens, including a results viewer, a
  review queue, a book chapter map and a dark theme.
- **Books are first-class**: front matter, a parsed contents page, per-chapter
  authorship and chapter-scoped reading.
- **Documents are classified before processing** and routed to a
  research-article, book or book-chapter pipeline; a whole folder is
  pre-flighted on the CPU first.
- **A permanent ledger**, keyed by file SHA-256, with a metadata review flow.
- **Local or cloud, chosen per role.**
- **Strategy Lab**: change which model does which job, with a diff before every
  save and a Restore Defaults.

### Fixed

- Figures on pages containing equations were silently discarded; fixing it
  recovered 22% more figures across the sample corpus.
- Reference lists full of boxed links were mistaken for figures; they are now
  told apart by how densely they are covered in text.
- `--force` now genuinely re-derives metadata instead of returning what the
  first run had stored.

### Licensing

Only OSI-permissive components ship. PyMuPDF (AGPL-3.0) left the shipping path
and DocLayout-YOLO (AGPL-3.0) was replaced by PP-DocLayoutV3 (Apache-2.0). **The
project licence became Apache-2.0.**

### Upgrading from 5.x

5.x remains available at the tags `v5.0.0` and `v5.2.0`. 6.0.0 is a rewrite
rather than an upgrade: the output format, the configuration layout and the
interface all changed, and there is no in-place migration. The differences
that matter first:

| | 5.x | 6.x |
|---|---|---|
| command line | `run_Prajnya.py` | `run_prajnya.py` |
| interface | `prajnya_studio/studio_app.py` | `run_prajnya_studio.py` |
| interface port | 8080 | **8850** (8080 is the model server's) |
| configuration | inside `prajnya_studio/` | one `config/` tree |
| output | beside the PDF | `run_outputs/` in the repository |
| ledger | none written | `persistence/semantic_ledger.csv` |
