# Changelog

What changed in each release of Prajnya. Versions are tagged in this
repository (`git tag`), and every output JSON records the version that produced
it in `document_info.pipeline_version`.

---

## 6.6.4

- **API keys no longer appear in the log.** The HTTP library under the Zotero
  client printed every request address at the normal log level, and Zotero's
  address contains your API key; `--verbose` could likewise print an OpenAlex
  key. These libraries now log only warnings and errors. **If you shared a
  Prajnya log made with a Zotero key, regenerate the key** (zotero.org,
  Settings, Security, Applications).
- **`online` is no longer described as unbuilt** in Studio's Settings and in
  `--help`. It has shipped since 6.6.0.
- Output is unchanged from 6.6.3.

---

## 6.6.3

- **The real example explains its Figure 1B.** `contract/examples/README.md`
  now says why the example's one table (`p6_tab1`) is panel B of Figure 1 and
  has an empty caption. Documentation only: the code and the output are
  unchanged from 6.6.1.

---

## 6.6.2

- **A real example of the output**, in
  `contract/examples/frontiers_hippo_hydra/`: one open-access paper (CC BY 4.0)
  processed end to end at citation level `online`, unedited. It holds the
  document JSON, the figure and table manifests and three crops; the
  attribution is in `contract/examples/README.md`. Documentation only: the
  code and the output are unchanged from 6.6.1.

---

## 6.6.1

- **Results screen: formulas and table headers display properly.** Table cells
  that hold LaTeX (`$R_{\mathrm{b}}(\mathrm{s}^{-1})$`) were shown as raw text;
  they are now drawn as formulas, offline, in both the table grids and the page
  text. A table's header row is marked as a header, a caption repeated as the
  first row is not shown twice, and grids get borders and padding. Display only:
  the stored JSON is unchanged.
- **README rewritten**, with screenshots of Prajnya Studio.

---

## 6.6.0

### New: reference lists, with every in-text citation linked

- **At citation level `online`, each paper gets its reference list**, in
  printed order: the publisher's deposited list from Crossref, gaps filled
  from OpenAlex, checked against the list printed in the PDF. Each entry
  carries its position, printed number, first author, year, title, journal,
  DOI and the text as printed.
- **Every citation printed in the paper's text is linked to the entry it
  names**, numbered (`[3–7]`, `(12, 14)`, superscripts) and author–year
  (`(Doi et al., 2005)`, `Berendsen (1962)`). Each one records the entries it
  cites and where it stands in the page text Prajnya writes out.
- **A wrong link is worse than none.** A numbered citation links only where
  an entry's printed number equals its place in the list; numbers beyond the
  list and affiliation marks on author lines are not citations. An
  author–year citation links only when exactly one entry has that surname and
  year; two candidates, or none, give no link.
- **`online` sends only the paper's DOI or title** to Crossref and OpenAlex,
  never page content. The default level is still `local`, which sends
  nothing. `--recite --citations online` adds reference lists and links to
  documents already processed, without reading their pages again.

### Changed: the citation level, and its default

- **One setting now decides how a paper's bibliographic record is
  established:** `citations.level` in
  `config/research_article_strategy.yaml`, with `--citations` on the command
  line to override it for one run, and a "Citation Level" choice on the
  Studio's Settings page.
  - `local` (**the new default**): Prajnya's own reading of the PDF. Nothing
    leaves the computer.
  - `zotero`: checked against your Zotero library, as before.
  - `online`: Crossref, by DOI or by title under a strict acceptance rule,
    with OpenAlex as fallback; the only level with reference lists and
    linked citations.
- **If you use Zotero, set `level: zotero`.** Until now Zotero was used
  whenever its credentials were present; from this release it is used only
  at that level. At `zotero` the results are unchanged (checked on the
  corpus). At `local` the title, authors and DOI are the same for every
  corpus document that prints a DOI, but the journal name is less reliable
  (`eval/citation_levels/GATE_S0B.md`).
- **`--recite` changes the level of documents already processed, without
  re-processing them.** It works out each document's record again from what
  its run already read: about two seconds for nine documents, with no PDF
  read and no model called. Pages, figures and tables are untouched, and a
  record corrected by hand is never replaced. A run that skips documents
  processed at another level now says so
  (`eval/citation_levels/GATE_S0C.md`).
- The old `zotero.enabled` and `zotero.offline_mode` switches are gone from
  the shipped config. A config file that still has them and names no level
  behaves as it did.
- A document's record of where its DOI came from (`doi_source`) is now kept
  in the output; it used to be dropped.

### Output format (contract 1.1)

- **Every document now says how to cite it.** A new top-level `citations`
  block gives the citation level the run used (`local`, `zotero` or
  `online`), which step supplied the metadata, and a short label to cite the
  document by ("Bella, 2016"). A document with no usable author, editor or
  title is cited by its file name, and says so; the file name is never
  written into the title. At `online` the block also carries the paper's
  reference list (`references`) and its linked in-text citations
  (`markers`, whose `numbers` are positions in `references`).
  This is an addition: programs written for contract 1.0 are unaffected.
  Two example files are in `contract/examples/`
  (`docs/citation_levels_plan.md`).

### Improved

- **A table continued onto the next page gets its column headers back.** When
  a table runs over two pages and the second page prints no header row, the
  header is taken from the first page and shown to the reader above the
  continuation, then removed from the result. It applies only when the two
  pages' column lines match exactly and the reader returns the header; any
  doubt reads the table as before. On the corpus it fixes the Springer book's
  Table 13.1 continuation (a dropped column put every value under the wrong
  gene) and improves Table 13.3's; the continuations that print their own
  header are unchanged, as are all other tables
  (`eval/header_carry/GATE.md`).
- **A figure the layout detector missed is recovered from its caption.**
  When a printed caption ("Figure 11 ...") has no figure cut out on its page,
  the space above that caption is searched again with a lower confidence
  threshold, and what is found there becomes the figure. Two corpus figures
  that were never extracted are now there, and a figure that had been given
  the wrong number from the previous page now carries its own
  (`eval/orphan_captions/GATE.md`).
- **A figure caption cut short is completed from the PDF's text** when the
  detected caption is only the figure's title, or stops partway: six article
  captions now carry their full legend (`eval/caption_source/GATE.md`).

---

## 6.5.0

### Setup

- **pypdfium2 is now 5.15.2, pinned exactly** (was `>=4.24,<5`). Reinstall
  from `requirements.txt`. Each pypdfium2 release bundles a different PDFium,
  and they extract text differently, so the version is fixed to the one the
  corpus was measured on.

### Improved

- **Figure captions end where the caption ends.** A caption read from the
  PDF's text now stops at the first line set in the article's body type
  (journals print captions in another font, or smaller). Captions that ran
  on into the article's prose now stop at their last sentence, and captions
  cut short by a stray symbol are complete. Measured on the corpus: every
  caption that changed is better (`eval/caption_endings/GATE.md`).
- **Text lines are grouped by their baseline**, not the top of each word's
  box, so a degree sign, a symbol or a bold label no longer splits a printed
  line in two.
- **Ligatures in the Springer book are read correctly** ("fibril", not
  "fbril"), from the newer PDFium.

### Fixed

- **A book's contents page is read under the new PDFium**, which no longer
  extracts the dots between a chapter title and its page number. The dots
  are now found as a page object (15 of 15 chapters found).
- **A figure label beside a narrow-gutter side caption is found**
  (Springer, "Fig. 2.3").
- **A figure panel whose detected caption was only its panel's sentence now
  gets its figure's label** when the caption is printed on the facing page.
  No figure in the corpus is left unlabelled.

### Known

- In the Springer book, a side caption printed beside the body column can
  still run into it (p70, p333). The book path is not yet a focus.
- The newer PDFium drops some spaces inside equations (`3 4hb` -> `34hb`).

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
