# The Prajnya output contract

Prajnya writes one JSON file per PDF (`run_outputs/documents/<name>.v6.json`),
plus cropped images of every figure and table. Other programs read these files.
This folder says exactly what they may rely on.

- **`prajnya_document.schema.json`** is the promise, as a JSON Schema
  (draft 2020-12). It is checked by `tests/test_output_contract.py` against
  the writer's output and against every document in `run_outputs/`.
- **Contract version: 1.6.** Every file written from v6.4.0 on says its
  version in `document_info.contract_version`. A file without that key was
  written by 6.3.0 or earlier and follows contract 1.0 in every respect.
- **`examples/`** holds two small hand-written documents that follow the
  contract, so a reader can be built and tested before real files exist:
  `citations_online.json` (a reference list and linked marks) and
  `citations_local_file_name.json` (a document cited by its file name).
  Their bibliographic content is illustrative. The tests validate both.
  `examples/frontiers_acidovorax_hrp/` is real, unedited output for one CC BY
  paper (see `examples/README.md`), also validated by the tests.

| version | first release | what changed |
|---|---|---|
| 1.0 | v6.4.0 | the contract is written down |
| 1.1 | v6.6.0 | the top-level `citations` block: how to cite the document, at which level that was established, and (level `online`) its reference list and linked in-text marks. Additive: a 1.0 reader is unaffected. |
| 1.2 | v6.7.0 | the top-level `mentions` block: in-text "Fig. 3B" / "Table 2" linked to the figure and table objects, at every level. Additive. |
| 1.3 | v6.7.0 | `pages[].body` (the page's running text), `pages[].metadata.furniture` (running headers, footers and page numbers, labelled) and `pages[].metadata.outside_body` (placeholders, tables, legends and the reference list, marked). Additive. |
| 1.4 | v6.8.0 | `metadata_final.author_names`: each author as `{given, family}`, in the order of `authors`, when a registry record gave the surname apart (Crossref at level `online`); `[]` otherwise. A compound surname ("Rosendahl Huber", "García Martínez de Artola") cannot be told from the joined string by any rule. Additive. |
| 1.5 | v6.8.0 | `document_info.status` (`success`, `partial`, `failed`) and `document_info.failed_pages`: whether every page was read. Before 1.5 a page whose reading failed was written empty and the file still said `success: true`. Additive; `success` keeps its meaning. |
| 1.6 | v6.9.0 | `document_info.provenance`: what produced the run (Python, OS, the PDF and imaging library versions, the PDFium build, the layout model file, each model role's model file and quantisation, the llama.cpp build, the GPU, driver and CUDA version). File names only, never paths. `null` when not recorded (an older file re-cited later). Also `pages[].metadata.display_formulas` on text-layer pages (rule 22). Additive. |

Anything the schema does not name is internal and may change in any release:
for example `pages[].metadata.analysis`, `coverage`, and the rest of the
`profile` block beyond `pdf_path` and `processing_result.chapters`.

## Version rules

| change | version |
|---|---|
| remove, rename or retype a key; change what a value means; add a value to a closed list that a reader may switch on (for example a new `processing_method`) | **MAJOR** (`1.0` to `2.0`) |
| add a key, or add a new optional record | **MINOR** (`1.0` to `1.1`) |

Readers must ignore keys they do not know. A new route or page type is listed
as MAJOR on purpose: a reader that maps routes to behaviour would otherwise
meet the new value silently. The tests refuse to let the code emit a value the
schema does not list.

A file is held to what its own `contract_version` promises: from 1.1 it
must carry `citations`, from 1.2 `mentions`, from 1.3 each page's `body`,
`furniture` and `outside_body`, from 1.4 `metadata_final.author_names`, from
1.5 `document_info.status` and `failed_pages`, from 1.6
`document_info.provenance` (which may be `null`). An older file, or one with no
version stamp, is not held to the later rules. A new key added in a MINOR
version goes into the schema's `allOf` list the same way, so a file that
claims the new version cannot leave it out.

When the contract changes, change the schema, `CONTRACT_VERSION` in
`src/processing/document_json.py`, and this file together. The test that
compares the first two fails if you forget.

## Things a reader must know that the shape alone does not say

1. **Two numbering systems.** `pages[].page_number` is **1-based**.
   `chapters[].page_range` is **0-based**, `[first, last]`; add 1 to each.
   It is `null` when a chapter's boundary was not found.
2. **Two coordinate systems.** Figure and table `bbox` are in **pixels** of the
   rendered page image (and equal the size of the saved crop). Heading `bbox`
   is in **PDF points**. Do not mix them.
3. **`processing_method` is what ran; `page_type` is what was predicted.**
   Where they disagree, believe `processing_method`. Two values mean the read
   **failed**: `glm_failed` and `two_shot_title_failed`. Such a page carries
   `metadata.error`, and its `content` is not a reliable reading. A reader
   should skip or flag these pages, not index them as text.
4. **`pymupdf` is the text-layer route's historical name.** The text layer is
   now read with pypdfium2. The label is kept so old and new files agree.
5. **Table `grid` is an HTML `<table>` string, and cells can contain LaTeX
   between `$...$`.** About 9 of the 31 corpus tables have such cells. Treat
   `$...$` as math markup, not prose. Cell text is a model's reading and can
   be wrong; the table crop (`image_path`) is the original.
6. **Figure captions can still be empty or wrong, though rarely** (in the
   111-figure corpus at v6.3.0: 1 empty, 2 not starting with a figure label).
   `printed_label` is null when no label was found (3 of 111).
7. **`image_path` is relative to `run_outputs/`**, with forward slashes.
8. **`profile.pdf_path` is an absolute path on the machine that ran Prajnya.**
   It is informational. Never open it, and never publish a file without
   considering that it names a directory on someone's computer. To share a
   file, write a portable copy, in which the field holds only the PDF's file
   name: `python -m src.processing.export <document>.v6.json`.
9. **A book's front-matter pages carry no `processing_method`** (a known,
   deferred gap). A reader should call their route "unknown", not guess.

10. **The `citations` block is present at every level** (from contract 1.1).
    A file with no block follows contract 1.0: treat it as carrying no
    reference list. Never infer the level from what is missing; read
    `citations.level`.
11. **`level` is what was allowed; `metadata_source` is what happened.** A
    run at `zotero` whose paper is not in the library has
    `metadata_source: extracted`.
12. **`source_label` is how to cite the document** ("Bella, 2016"). When
    `cited_by_file_name` is true it is the PDF's file name, and
    `metadata_final.title` is null: the file name is never written into
    the title. Such a document needs a human.
13. **`reference_list.status` separates three kinds of "no list"**:
    `not_requested` (the level does not fetch one), `not_available` (asked;
    none exists) and `not_reached` (the network failed, nothing is known).
14. **A reference is a cited work, not a read one.** Prajnya has not read
    the paper behind `references[]`. A sentence resting on it should be
    cited as "Doi et al., 2005, cited in Bella, 2016", never as the primary
    paper alone.
15. **`markers` can be empty while `references` is not.** Marks are linked
    only when the list passed its checks; a wrong link is worse than none.
    A mark's `numbers` are the `position`s of the references it cites (for a
    numbered list a linked reference's printed number equals its position;
    an author-year mark, `kind` "author_year", is linked by author and year
    and only when exactly one entry fits).
    `content_offset` indexes that page's `content` string and is null when
    the mark could not be found there exactly once.
16. **`mentions` links "Fig. 3B" to a figure by its printed label, not by
    reading the figure** (from contract 1.2). Supplementary mentions ("Figure
    S1", "S1 Fig", "Figure 2-figure supplement 1") are never linked to a main
    figure; they are only counted in `skipped`. A mention whose `targets` is
    empty names a figure or table Prajnya did not find. A mention is found in
    `content` as written, so a mention inside text Prajnya misread is missed.
17. **`body` is for passages; `content` is the record** (from contract 1.3).
    `body` is the page's running text: `content` without the furniture
    (`metadata.furniture`: running headers, footers, page numbers) and
    without the parts in `metadata.outside_body` (Prajnya's placeholders,
    tables, legends, the reference list). Nothing is deleted from
    `content`, and each part left out is kept elsewhere in the file or is
    Prajnya's own markup. Every offset in the file indexes `content`, never
    `body`. All of it is found from the text: a legend merged into a
    sentence by a model read, or a reference list with no "References"
    heading in the text, stays in `body`; footnotes, affiliations,
    acknowledgements and other back matter are not taken out.
18. **Use `author_names` for surnames when it is not empty** (from contract
    1.4). `authors` holds each name as one string ("Sietske K. Rosendahl
    Huber"), and no rule can find where a compound surname begins.
    `author_names[i]` is the same person as `authors[i]`, with `family`
    exactly as it ends that string. It is filled only from a registry record
    that keeps the surname apart (Crossref, at level `online`); otherwise it
    is `[]`, and splitting `authors` is a guess.
19. **Check `status`, not only `success`** (from contract 1.5). A document
    can say `success: true` and still have lost pages: `success` only means
    the strategy raised no error. `status` is `"partial"` when some pages
    failed; `failed_pages` lists them. A failed page keeps its text layer
    when it has one, but its figures and tables are missing, so a reader
    building an index should flag or skip those pages.
20. **`quality_score` is a rough guess, not a measured accuracy.** On a
    model-read page it combines text length, Markdown structure, odd
    characters and whether the expected fields came back; some routes set a
    fixed value (0.95 for a text-layer page, 0.0 for a failed one). Nothing
    in Prajnya acts on it. Do not filter or rank pages by it: use
    `processing_method` and `status` to tell a failed read from a good one.
21. **Compare runs by `provenance`, not by `pipeline_version` alone** (from
    contract 1.6). The same release can read a PDF differently with another
    model file, quantisation, llama.cpp build or library version. `models`
    lists the model configured for each role; which pages actually used a
    model is in each page's `processing_method`. A part that could not be
    read on that machine (no GPU, the model server not running) is `null`.
22. **A text-layer page can hold LaTeX** (from v6.9, contract 1.6). On a
    page read from the text layer (`processing_method: pymupdf`), each
    display equation the layout model finds is read from its crop and
    written as `$$...$$` in place of the text layer's (often garbled) run.
    `metadata.display_formulas` lists every box found, `bbox` in **PDF
    points** (like headings), with `status`: `read` (replaced),
    `not_in_text` (its words were not one unbroken run; left as printed) or
    `empty_read` (the read gave nothing; left as printed). Absent when the
    page has none. Inline formulas stay as the text layer prints them.

## Checking a file yourself

```
python -c "import json, jsonschema; \
s = json.load(open('contract/prajnya_document.schema.json', encoding='utf-8')); \
d = json.load(open('run_outputs/documents/<name>.v6.json', encoding='utf-8')); \
jsonschema.validate(d, s); print('valid')"
```
