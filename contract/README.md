# The Prajnya output contract

Prajnya writes one JSON file per PDF (`run_outputs/documents/<name>.v6.json`),
plus cropped images of every figure and table. Other programs read these files.
This folder says exactly what they may rely on.

- **`prajnya_document.schema.json`** is the promise, as a JSON Schema
  (draft 2020-12). It is checked by `tests/test_output_contract.py` against
  the writer's output and against every document in `run_outputs/`.
- **Contract version: 1.0.** Every file written from v6.4.0 on says so in
  `document_info.contract_version`. A file without that key was written by
  6.3.0 or earlier and follows contract 1.0 in every respect.

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
   considering that it names a directory on someone's computer.
9. **A book's front-matter pages carry no `processing_method`** (a known,
   deferred gap). A reader should call their route "unknown", not guess.

## Checking a file yourself

```
python -c "import json, jsonschema; \
s = json.load(open('contract/prajnya_document.schema.json', encoding='utf-8')); \
d = json.load(open('run_outputs/documents/<name>.v6.json', encoding='utf-8')); \
jsonschema.validate(d, s); print('valid')"
```
