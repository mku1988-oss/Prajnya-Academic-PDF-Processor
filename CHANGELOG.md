# Changelog

What changed in each release of Prajnya. Versions are tagged in this
repository (`git tag`), and every output JSON records the version that produced
it in `document_info.pipeline_version`.

---

## 6.4.0

### Setup

- **Tested with llama.cpp build b12035.** llama.cpp 1.0.0 (13 November 2026)
  rejects some vision-projector files at load that earlier builds accepted
  (llama.cpp PRs #32129 and #32130). Prajnya's three OCR projectors were
  checked against those rules on file and appear to comply, but they have not
  yet been loaded on 1.0.0; until then, stay on a build before it.

- **The model server's default port is now 9931, not 8080.** llama-server
  has used 9931 as its own default since build b11541 (10 October 2026), so a
  server started without `--port` was no longer where Prajnya looked.
  `run_llama_server.py` and `config/research_article_strategy.yaml` both
  change, and a test now checks that the two agree. **If you start
  llama-server yourself with `--port 8080`, either drop that flag or set
  `vlm_provider.local_server.port` and `api_endpoint` back to 8080.**

### Fixed

- **A fresh install of 6.3.0 could not process any page.** The requirements
  allowed any pypdfium2 from 4.24 up, so a new install received 5.x, which
  renamed a method the page classifier calls on every page. Prajnya now works
  with either name, and pypdfium2 is pinned below 5: version 5's newer PDFium
  also reads text differently (on a book's contents page it drops the dots
  and the space before each page number, so "Life . . . 45" becomes
  "Life45"), which needs measuring before it is adopted.

- **Word positions from the PDF are now exact.** The text layer's word boxes
  were estimated by dividing each run of text evenly among its words, which
  erased the gaps between them and, where runs overlapped, repeated characters
  (1,922 on 240 of the corpus's 505 pages: "NiNi" for "Ni", "fifi" for the fi
  ligature). They are now built from each character's own box, and match
  PyMuPDF's exact boxes closely (median overlap 0.90, was 0.66). Caption text,
  figure labels and region text read from the page no longer carry doubled
  characters.
- **Figure captions beside a column of prose are found again.** The caption
  scan split a line into two columns only at a gap of 6% of the page width,
  wider than a real two-column gutter, so on some pages a caption ran into the
  neighbouring column's prose and its label was missed. The split is now
  measured in text heights. Corpus: 3 figure labels gained, none lost.
- **More figure captions are read in full.** Rotated text in a page margin
  (a journal's "Downloaded from ..." stamp) and lines from the neighbouring
  column no longer cut a caption short. On the corpus, 13 captions that used
  to stop after their first line are now read further. Known issue: on 3 of
  them the caption runs on into the first lines of the next paragraph, and a
  caption printed in two columns can come back with its lines interleaved.
- **Short words are no longer mistaken for rotated text.** Whether a word is
  rotated now comes from the PDF's own character angles instead of the shape
  of its box, which dropped words such as "of" and sent 26 prose pages to
  the slower model read for no reason. The corpus run is 3% faster than 6.3.0.
- **The check that sends a page for a table re-read is honest again.** It
  credited a "missing" word if the word appeared anywhere inside another word
  of the reading, so a page could lose 40% of its words and still pass. It
  now credits only real ligature and hyphenation splits (including PDFium's
  own line-end hyphen and a ligature that kept only its first letter).

### Output format

- **The output JSON now has a written contract.** `contract/` holds a JSON
  Schema and a one-page explanation of what a program reading our files may
  rely on, and `document_info.contract_version` (currently `1.0`) says which
  version a file follows. Nothing about the existing shape changed: the schema
  describes the files 6.3.0 already wrote, and a test checks it against all of
  them. Things the shape alone did not say are now written down, among them
  that chapter `page_range` counts from zero while `page_number` counts from
  one, that figure boxes are in pixels but heading boxes are in PDF points,
  and that two `processing_method` values mean the page's read **failed**.

---

## 6.3.0

### Tables

- **Tables are read from the table itself, the right way up.** The reading
  model is now given only the table body, without its caption and footnote,
  and a table printed sideways on the page is turned upright before it is
  read. Sideways tables are recognised from the PDF's own text, so nothing
  has to be marked by hand. The saved table image and its caption are
  unchanged. On the sample corpus this corrected 4 of the 7 tables that had
  been read seriously wrong (previously 1), and on 20 tables picked at random
  from a reference library it made none worse; on both, the text read from
  tables matches the printed table more closely, and table reading is about
  20% faster.
- **Known trade-off:** the transcription can be slightly less literal. On one
  long table, printed "...." no-data markers come out as empty cells, and two
  single-character misreads were seen ("1½" read as "1%", "Drosophila" as
  "Brosophila") that the previous crop read correctly.

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
