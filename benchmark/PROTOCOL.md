# Prajnya benchmark: protocol

**Status:** FROZEN 2026-11-19 (session 39), approved by the researcher before
any tool was run on the corpus. Every later change goes in
**Amendments** at the bottom, with its date and reason, and is never edited
silently.

**Purpose:** measure how Prajnya and other open tools turn scientific PDFs into
structured data, on papers whose correct answer is published by the
publisher, so that nobody hand-labels anything and nobody grades their own
homework. Results live in `benchmark/` for interested readers. They are not
put in the README.

---

## 1. Ground rules

1. **This protocol is written and committed before any tool is run on the
   scored corpus.** Metrics, normalisation and pass/fail rules do not change
   after results are seen. A defect found later in the harness (a bug in our
   code, not a score we dislike) is fixed as an amendment, and every tool is
   re-scored.
2. **Prajnya is frozen** for the scored run: one tagged version (§4), no code
   or config change prompted by benchmark results until the results are
   published. Fixes go to the next release and a later re-run.
3. **Every tool runs as its maintainers ship it**: pinned version, documented
   default settings, the recommended model set. No tuning of any tool on the
   scored corpus, Prajnya included.
4. **A field a tool does not produce is "not supported", never zero.** It is
   shown as `n/s` and left out of that tool's averages.
5. **Everything is published:** harness, adapters, tool settings, per-paper
   scores and raw outputs. Tool maintainers are invited to correct our
   settings, and a correction leads to a re-run that is recorded in the
   amendments.
6. **Losses are published with the wins.** No result is dropped.

## 2. Answer key: publisher JATS XML

Each paper comes from the **PMC Open Access subset** with both its PDF and its
publisher's JATS XML (the structured version of the same article). The XML is
the answer key:

| what is scored | where it is in JATS |
|---|---|
| title, authors, DOI, year, journal | `article-meta` (`article-title`, `contrib[@contrib-type="author"]`, `article-id[@pub-id-type="doi"]`, `pub-date`), `journal-meta/journal-title` |
| abstract | `article-meta/abstract` (the first, untyped one) |
| body text and its order | `body//p`, in document order, under `sec` |
| figures | `fig`: `label`, `caption` |
| tables | `table-wrap`: `label`, `caption`, `table` (cells) |
| reference list | `back/ref-list/ref`: `element-citation` / `mixed-citation` |
| in-text citations | `xref[@ref-type="bibr"]/@rid` → `ref/@id` |
| in-text figure mentions | `xref[@ref-type="fig"]/@rid` → `fig/@id` |

**What JATS cannot key, and how it is handled:**

- **Figure crops and positions.** JATS has no boxes. Figures are scored by
  label and caption only, never by crop quality.
- **Panels.** JATS marks a figure, not its panels. "Panels kept together" is
  scored as *one output object per printed figure label* (§6.3).
- **Tables given only as an image** (`table-wrap` with a `graphic` and no
  `table`) have no cells to compare. They count for table *finding* but are
  left out of cell scoring. How many were left out is reported.
- **Text the PDF and the XML print differently** (hyphenation, ligatures,
  display formulas, footnotes) is handled by the normalisation in §5, the same
  for every tool.
- **Supplementary material** is out of scope on both sides.

The key is built by `benchmark/ground_truth/jats_to_common.py`, which writes
the common format (§3). It is unit-tested, and its output on three papers is
checked by eye against the PDFs before any tool is scored.

## 3. Common format

One adapter per tool converts its output to one JSON file per paper. The answer
key uses the same shape. A field is `null` when the tool does not produce it
(→ `n/s`); an empty list means the tool produces the field and found nothing.

```json
{
  "paper_id": "PMC1234567",
  "tool": "prajnya", "tool_version": "v6.6.4", "track": "offline",
  "metadata": {"title": "...", "authors": [{"surname": "...", "given": "..."}],
               "doi": "10....", "year": 2021, "journal": "..."},
  "abstract": "...",
  "body_paragraphs": ["...", "..."],
  "figures": [{"id": "...", "label": "Figure 3", "label_number": "3",
               "caption": "...", "object_count": 1}],
  "tables":  [{"id": "...", "label": "Table 1", "label_number": "1",
               "caption": "...", "html": "<table>...</table>"}],
  "references": [{"id": "...", "position": 1, "title": "...", "year": 2005,
                  "doi": "10....", "first_author_surname": "...", "raw": "..."}],
  "citation_links": [{"paragraph_index": 12, "reference_ids": ["..."]}],
  "figure_links":   [{"paragraph_index": 12, "figure_ids": ["..."]}]
}
```

`citation_links` and `figure_links` hold each in-text mention as the set of
references (or figures) it names, placed in the paragraph where it occurs.
A range such as "[3–7]" is expanded to the five references.

## 4. Tools and tracks

| tool | version | how it is run | network |
|---|---|---|---|
| **Prajnya** | v6.6.4 (tag), llama.cpp b12035, released `config/` | `run_prajnya.py --noninteractive`, models per the released YAML | `local`: none |
| **Prajnya `online`** | same | `--citations online` | sends only the DOI or title to Crossref / OpenAlex |
| **GROBID** | 0.9.1, image `grobid/grobid:0.9.1-full`, CPU | `/api/processFulltextDocument`, `consolidateHeader=0`, `consolidateCitations=0`, `teiCoordinates=figure` | none |
| **GROBID consolidated** | same | `consolidateHeader=1`, `consolidateCitations=1` (Crossref) | sends extracted fields to Crossref |
| **Docling** | pinned at setup (C1) | default `DocumentConverter`, default pipeline and models | none |
| **Marker** | pinned at setup (C1) | default settings, no LLM option | none |
| **MinerU** | pinned at setup (C1) | default backend, and the VLM backend as a second row | none |

- **Versions and every non-default flag** go in `benchmark/tools.lock.md`,
  committed before the pilot (C4). Each tool lives in its own environment
  (conda env or Docker image), never in Prajnya's `requirements.txt`.
  MinerU (AGPL) and Marker (GPL) are run, not depended on.
- **Two tracks.** Results tables list every row above. A tool that uses the
  network is marked as such in the `network` column; nothing is hidden or
  scored separately because of it. The offline rows can be compared with each
  other as a set.
- **NotebookLM is out of scope.** It is a chat tool and produces no structured
  output to score.

## 5. Normalisation (the same for every tool and for the key)

For every text comparison:

1. Unicode NFKC; ligatures expanded (`ﬁ` → `fi`).
2. Every dash and minus sign mapped to `-`; every quotation mark to `"` or `'`.
3. A hyphen at a line end followed by a lowercase letter is joined
   ("differ-\nentiation" → "differentiation").
4. Whitespace collapsed to single spaces; trimmed.
5. Markup removed (Markdown, HTML, LaTeX commands); the text inside kept.
6. Display formulas left out of body text on both sides (they are not
   scored in this version).
7. For *matching* (finding which output item corresponds to which key item)
   text is also lowercased and stripped of punctuation. For *scoring* text,
   steps 1–6 only.

Labels are reduced to a kind and a number: "FIGURE 3 |", "Fig. 3", "Figure 3."
→ (`figure`, `3`); "Table S2" → supplementary, out of scope.

## 6. Metrics

All scores are computed per paper, then averaged over papers (macro), with the
pooled count (micro) given beside where it differs. Every metric reports its
per-paper distribution, not only the mean.

### 6.1 Metadata
- **title**: exact match after §5 matching normalisation (yes/no), plus
  character similarity (1 − normalised Levenshtein distance).
- **authors**: surname list F1, and whether the order is right.
- **DOI**: exact, case-insensitive, bare form.
- **year**: exact, against the earliest of `pub-date` epub / ppub /
  collection.
- **journal**: exact against `journal-title` or any `abbrev-journal-title`.

### 6.2 Abstract and body text
- **abstract**: character similarity.
- **body coverage**: share of key paragraphs matched by an output paragraph
  span with similarity ≥ 0.8 (output paragraphs may be merged or split; the
  matcher aligns on character spans, not paragraph boundaries).
- **body similarity**: character similarity of the full body text, key vs
  output, in the output's order.
- **reading order**: Kendall's tau between the key order and the output
  order of the matched paragraphs.

### 6.3 Figures
- **found**: figure-label recall and precision (by `label_number`).
- **caption**: character similarity of the caption, for each found figure.
- **kept together**: share of key figures that map to exactly one output
  object (a figure split into two crops, or two figures merged into one,
  fails).

### 6.4 Tables
- **found**: table-label recall and precision.
- **TEDS** (Tree-Edit-Distance-based Similarity, the standard table metric
  from PubTabNet) on structure and content, and **TEDS-S** on structure only,
  for every found table with cells in the key.
- Image-only key tables: counted for *found*, not for TEDS (§2).

### 6.5 References
- **matching**: an output entry matches a key entry by DOI; failing that, by
  title similarity ≥ 0.9 and the same year; failing that, by raw-text
  similarity ≥ 0.8. One-to-one, best match first.
- **recall** and **precision** of entries.
- **per field**, over matched entries: title (similarity ≥ 0.9 counts as
  right), year, DOI, first-author surname.

### 6.6 In-text citations
- For each key paragraph, the multiset of references cited in it (from
  `xref[@ref-type="bibr"]`, ranges expanded) is compared with the multiset
  the tool's links place in the aligned output paragraph(s), after mapping
  output reference IDs to key IDs by §6.5's matching.
- **link precision**: correct links / links the tool made.
- **link recall**: correct links / citations in the key.
- A wrong link counts against precision; a missing link only against recall.

### 6.7 In-text figure mentions
- As §6.6, with `xref[@ref-type="fig"]` and figure IDs.

### 6.8 Determinism
- Every tool runs the whole corpus **twice**, under the same conditions, with
  a restart of the tool's server (and llama-server) between runs.
- Reported: share of papers whose common-format output is byte-identical, and
  for the others, which fields changed.

### 6.9 Cost (reported, not ranked)
- wall time per page (whole corpus ÷ pages), and per paper;
- peak VRAM (`nvidia-smi`, sampled every second) and peak host RAM;
- size of the models used, on disk and in parameters as published;
- hardware: one machine, stated in full (CPU, RAM, GPU 8 GB), Windows 11.

**Fair-timing rule:** each tool runs alone. Before each tool: no other model
server running, all llama-server models unloaded, VRAM at its idle baseline,
which is recorded. Long runs are started detached.

## 7. Corpus

**Size:** 40 scored papers, plus 10 pilot papers drawn separately. Pilot
papers are never scored in the published results.

**Source and licence:** PMC Open Access subset, **CC BY only** (CC BY 4.0 or
3.0), so that PDFs, outputs and scores can be redistributed with attribution.
Each paper's licence is read from its own JATS `license` element and recorded.

**Strata (8 papers each):** Frontiers, PLOS, eLife, Nature Communications,
Cell Press open access (Cell Reports, iScience and similar). Within the 40,
at least:
- 15 papers with a two-column layout and 15 with one column;
- 15 papers with at least one table with cells in JATS;
- 15 papers with at least one figure of three or more panels;
- 10 papers with numbered citations and 10 with author-year;
- a reference list of 60 or more entries in at least 10 papers.

**Field:** biology (life-science subject terms), published 2018–2025.

**Exclusions, applied before the draw:**
- any DOI already in Prajnya's ledger, `run_outputs/`, any `eval/` key or
  `contract/examples/` (this removes the Frontiers Hydra paper);
- papers whose JATS has no `body` (PDF-only deposits), and retracted papers;
- PDFs with no text layer (should not occur in this source; checked).

**Draw:** candidates are listed by a script that queries PMC's OA service per
stratum. Papers are drawn at random with a **seed fixed in this protocol
before the listing is made**: seed `20261119`. If a stratum's quotas cannot
be met, the draw continues in list order and the shortfall is reported. The
list of candidates, the seed and the chosen papers are committed in
`benchmark/corpus/MANIFEST.tsv` (PMCID, DOI, publisher, licence, pages,
strata flags) before any tool runs.

PDFs and JATS files are stored in `benchmark/corpus/` (gitignored) and
re-fetchable from the manifest.

## 8. Phases and their gates

| phase | what | done when |
|---|---|---|
| **C0** | this protocol | committed with "PROTOCOL FROZEN" |
| **C1** | tools installed, versions in `tools.lock.md`, smoke test of each on one non-corpus PDF | every tool returns output |
| **C2** | `jats_to_common.py` + unit tests; key checked by eye on 3 pilot papers | tests pass; eye check noted |
| **C3** | one adapter per tool + unit tests | each adapter maps its smoke output |
| **C4** | **pilot**: 10 pilot papers, Prajnya and GROBID, all metrics | the harness runs end to end; defects found are fixed as amendments. Pilot numbers are not published. |
| **C5** | **full run**: 40 papers, every tool, twice | all outputs present; cost logs complete |
| **C6** | `benchmark/RESULTS.md`: every metric, every tool, distributions, losses, and the conditions | read by the researcher before anything is pushed |

## 9. What this benchmark does not claim

- It measures **one machine** and **40 biology papers** from five publishers.
  It says nothing about other fields, scans or books.
- Figure **crops** are not scored, only labels and captions.
- Formulas are not scored.
- Retrieval quality downstream (RAG) is not measured here.

---

## Amendments

### A1 (2026-11-19, before the draw; no tool had been run on any candidate)

How §7 is carried out, settled while writing the corpus script:

1. **Source.** PMC retired its OA Web Service in August 2026. Papers come from
   PMC's public open-data bucket `s3://pmc-oa-opendata` (per article: a JSON
   record with `license_code`, `is_retracted`, `doi`; the PDF; the JATS XML).
   Candidates are listed with NCBI E-utilities (`db=pmc`,
   `"cc by license"[filter]`, `2018:2025[pdat]`), one query per journal and
   year. The highest version present in the bucket is used.
2. **Journals per stratum:** Frontiers = Front Cell Dev Biol, Front Genet,
   Front Physiol, Front Microbiol, Front Immunol; PLOS = PLoS Biol, PLoS Genet,
   PLoS Pathog; eLife = eLife; Nature Communications = Nat Commun; Cell Press =
   Cell Rep.
3. **Biology.** These journals are life-science journals except Nature
   Communications, whose PMC XML carries no subject codes. For it, a paper is
   eligible only if its PubMed record carries at least one of the MeSH
   headings Animals, Humans, Bacteria, Plants, Viruses, Fungi, Cells Cultured.
4. **Research articles only:** JATS `article-type="research-article"`.
   Corrections, reviews, editorials and the like are not eligible.
5. **Licence:** `license_code` in the bucket's JSON must be exactly `CC BY`.
6. **Quota rule.** Each stratum's candidates are shuffled once with seed
   `20261119`; the first 8 eligible papers are taken. If a §7 minimum is then
   unmet, the script walks forward in each stratum's shuffled list and swaps
   the latest-drawn paper that lacks the feature for the next eligible paper
   that has it, stratum by stratum, until the minimum is met or the lists run
   out. Then the next 2 eligible papers per stratum are the pilot set.
7. **Feature flags** are read from the JATS: tables with cells
   (`table-wrap//table`); a figure of three or more panels (three or more
   distinct panel letters in one caption, as `(A)`, `A,` or a bold letter);
   citation style (`xref[@ref-type="bibr"]` text all digits/ranges → numbered,
   otherwise author-year); reference count (`ref-list/ref`). Column layout is
   read from the PDF (page 2: share of text lines starting right of the page
   centre ≥ 0.25 → two-column).

### A9 (2026-11-21) ⚠️ made AFTER run 2's scores had been seen: Prajnya's surnames split wrongly by the adapter

**What was wrong (harness, not tool).** Prajnya writes each author as one
string; its adapter took the last whitespace word as the surname. Surnames
with a particle were cut ("Dennis van der Meer" -> "Meer", "Karen Perez de
Arce" -> "Arce"), and an affiliation mark printed after a name ("Giulia
Zanetti {,}", "Torbjørn Elvsåshagen,,✉") became the "surname" and folded to
nothing. Found while reading Prajnya's wrong author lists on run 2 for the
report's definitions.

**What changes.** `adapters/prajnya.py` `split_name`: the surname is the last
word that has letters, with the particles before it (`SURNAME_PARTICLES`).
Misreadings in Prajnya's output ("Gonc¸ alves", "Galichotte") are untouched.
Runs 2 and 3 were re-scored for every tool, on both keys.

**Effect (run 2, scored).** Only `authors_f1` and `authors_order` moved, and
only for Prajnya: local 0.912 -> 0.945 and 0.625 -> 0.650; online 0.918 ->
0.952 and 0.625 -> 0.650. No other metric or tool changed.

### A8 (2026-11-20, 19:05, written and committed BEFORE the draw it describes) a fresh scored set

**Why.** Prajnya's fixes for item 000a (fixes 1-6) were built on the pilots,
but D1 records that scored papers' text was seen while building fix 1, and
run 1's scored results had been seen before every fix. Scoring Prajnya again
on the same 40 would reward fitting to them. The researcher asked for papers
nobody has looked at.

**What changes.**
1. **A new scored set of 40**, drawn by §7 and A1 (with A3) exactly as
   before, from the same cached candidate listings (19 November), with a new
   seed fixed here before the draw: **`20261120`**.
2. **Excluded** in addition to §7's exclusions: all 50 papers of the first
   draw (every DOI in the current manifest).
3. **No new pilots.** The 10 pilots stay the development set.
4. **The first 40 become role `seen`.** They are kept in the manifest and
   may be reported as a separate, non-blind set; they are never pooled with
   the new 40.
5. **Run 2** runs every tool on the new 40 (and the pilots). Prajnya's code
   is frozen before run 2 starts; its commit is recorded in the run's log.
6. **Blindness.** Until run 2's scores are written, no development script
   reads a `scored` paper, and nobody opens their PDFs, XML or outputs.
   Scripts select papers through the manifest's `role`.

Command: `build_manifest.py --a8 --data-root <development checkout>`; report
`corpus/DRAW_REPORT_A8.json`.

### A7 (2026-11-20) ⚠️ made AFTER scored results had been seen: answer-key defect, Prajnya's new fields, table mentions

**What was wrong (answer key).** §2's paragraph rule skipped any `<p>`
containing another `<p>` as a "wrapper". Cell Press and Nature Communications
JATS anchor a `<fig>` (whose caption has its own `<p>`) or a `<table-wrap>`
inside the paragraph that first cites it, so that paragraph, the very one
naming the figure, was dropped from the key with all its links. It was found
on the pilots (item 000a, fix 1). Conversely, a `<table-wrap>` with no inner
`<p>` (Cell Press's Key Resources Table) stayed in its paragraph, so a whole
table counted as body text.

**What changes.**
1. **Key.** A `<p>` is a wrapper only when a nested `<p>` is itself body text
   (not inside a figure, table, caption, ...). A figure or table anchored in a
   paragraph is not paragraph text, and the cross-references inside it are not
   the paragraph's links.
2. **Table mentions (new metric).** `xref[@ref-type="table"]`, scored exactly
   as §6.7's figure mentions: `table_mention_precision`, `table_mention_recall`.
   GROBID's `<ref type="table">` are read by its adapter.
3. **Prajnya's adapter** reads what contract 1.2 and 1.3 outputs carry: the
   `mentions` block becomes figure and table links; the spans each page marks
   as not running text (`metadata.furniture`, `metadata.outside_body`) are
   blanked like the headings and tables already were. Older outputs convert as
   before.

**How it is reported.** Every metric is reported on the amended key; the
frozen key (`outputs/key_frozen/`) is scored beside it for run 1 and run 2
(`summary_<role>_key_frozen.json`). Run 1's recorded `summary_pilot.json` and
`summary_scored.json` are the frozen-key scores and are not rewritten.

### D1 (2026-11-20) ⚠️ protocol DEVIATION, not a rule change: scored papers' text seen during development

**What happened.** §1 keeps development on the 10 pilot papers. While
building Prajnya's page-furniture rule (item 000a, fix 1, branch
`feature/weak-spots`), the survey read run 1's stored Prajnya outputs from
`outputs/run1/prajnya/rundir/documents`, which holds all 50 papers, not only
the pilots. The first and last lines of every page of the 40 SCORED papers
were seen, and the lines the draft rule labelled on them.

**What it influenced.** Three choices of that rule were partly informed by
scored papers: lines starting with an object label (Figure, Table, Appendix)
never count; every line of a folio shape must agree on one page offset; the
repeat threshold was kept rather than lowered (its false positives on scored
papers were seen). Fixes 2 to 6 were built on the pilots only. The scores
used to judge fix 1 were computed on the pilots only.

**What it means for run 2.** Prajnya's body coverage, body similarity and
link placement on the scored set depend in part on that rule, so they are
not blind for it. Every other metric is unaffected. Run 2's report states
this deviation beside those three metrics. No score was looked at.

**Guard.** Development scripts select papers through the manifest's
`role == "pilot"`, never by listing an outputs folder.

### A6 (2026-11-20, 08:40) ⚠️ made AFTER scored results had been seen

**What was wrong.** §6.5's third matching rule compares raw reference text by
plain similarity. For publishers that deposit structured citations
(`element-citation`: Nature Communications, Cell Reports, eLife), the key's
"raw" text is rebuilt from fields: no "et al.", the year moved, PMIDs added.
A tool that returns references as printed text only (MinerU) then matched
0.13 of Nature Communications references although a word-order-free check
finds a MinerU entry for 904 of 905 Nature Communications and Cell Reports
references.

**What changed.** The raw-text rule compares word sets (rapidfuzz
`token_set_ratio` ≥ 0.8) instead of plain similarity. DOI and title rules are
unchanged; this rule matters only where they fail. Applies to every tool, and
to the reference matching behind §6.6's citation links.

**How it is reported.** `references_recall` uses this rule;
`references_recall_frozen_rule` is the frozen rule, published beside it.

### A5 (2026-11-20, 08:30) ⚠️ made AFTER scored results for Prajnya, GROBID, Docling and MinerU had been seen

**What was wrong.** §6.3 matches figures by printed label only. MinerU keeps a
figure's caption but separates the label line ("FIGURE 6") from it, so a
figure MinerU found and captioned correctly scored as not found. This is a
defect of the rule, not a property of the tool: the protocol's intent is to
ask whether the figure was found.

**What changed.** A key figure not found by label is also matched, one to
one, to an *unlabelled* tool figure whose caption (label stripped) is ≥ 0.8
similar to the key caption. This applies to every tool.

**How it is reported.** Both numbers are published side by side:
`figures_recall` (the frozen rule) and `figures_recall_with_caption_match`
(this amendment); likewise for kept-together. Neither replaces the other.
Effect on run 1 (40 scored papers): Prajnya 0.940 → 0.940; GROBID 0.829 →
0.891; Docling 0.693 → 0.731; MinerU 0.546 → 0.724; Marker scored after.

### A4 (2026-11-20, harness built on pilot papers and on the excluded Frontiers Hydra paper; no scored paper had been looked at)

How the metrics are computed, where §6 left a choice:

1. **Figure object (§6.3).** The unit each tool itself presents as one
   figure: a GROBID `<figure>`; a Docling picture (its caption = its
   `captions` list or caption items whose `parent` is the picture); a Marker
   FigureGroup/PictureGroup, or a Figure/Picture outside any group; a MinerU
   image or chart block; a Prajnya label group (all crops sharing one
   `label_key`). An object whose caption carries no "Figure N" label is
   counted as *unlabelled* and reported separately.
2. **eLife figure supplements** (`fig[@specific-use="child-fig"]`, labels such
   as "Figure 2—figure supplement 1") are supplementary (§2) and leave the key;
   a tool's object with such a label is not a main figure.
3. **Invisible characters** (zero-width spaces, word joiners, soft hyphens)
   are deleted in normalisation (§5 step 1), for every tool and the key: the
   JATS of primer sequences carries zero-width spaces the PDF does not.
4. **Body alignment (§6.2).** Each key paragraph is searched for in the tool's
   whole running text (word 4-gram seeds, 2-gram seeds for paragraphs under 6
   words; then a fuzzy local alignment). The aligned span's similarity is the
   paragraph's score. Running text excludes what the tool itself marks as
   heading, caption, header/footer, table or figure; for Prajnya, whose page
   text is unlabelled, the Markdown headings, embedded tables and images and
   the headings in page metadata are blanked.
5. **Link placement (§6.6-6.7).** A link is placed by its character position:
   it belongs to the key paragraph whose aligned span (score ≥ 0.5) contains
   it, or lies within 40 characters of it; otherwise it is unplaced and
   counts only against precision. Prajnya markers without a `content_offset`
   are placed by their `context_before` text.
6. **References a tool lists but does not parse** (MinerU's `ref_text` list)
   are matched on raw text (§6.5, third rule) and count for recall and
   precision; their per-field scores are `n/s`.
7. **A paper a tool fails on** (no output, or a crash) is scored as an empty
   output in every field that tool supports.
8. **§5 step 3 in the alignment.** The alignment stream joins a word broken
   by a hyphen at a line end before a lowercase letter, as §5 requires, while
   keeping every character's original position for link placement.
9. **Baseline row.** The PDF's raw text layer (pypdfium2, no tool) is aligned
   against the key the same way and reported as a reference row for body
   coverage, so readers can see what a tool adds over plain extraction. It
   also checks the key: a paper whose JATS did not match its PDF would show
   up here (none did: median 0.90, minimum 0.63, all misses traced to
   captions, footers and inline maths interrupting the text layer).
10. **Caption similarity** compares captions without a leading label
    ("Figure 3 |", "Fig. 3."): the key's caption never contains it, and the
    label is scored separately (§6.3 found).
11. **Table precision** is over labelled tool tables, as for figures; tables
    the tool gives without a "Table N" label (appendix tables, continuation
    pieces) are reported as `tables_unlabelled`, not counted as errors.
12. **LaTeX symbols** in tool output (`\alpha`, `\pm`, `^{\circ}`) become their
    Unicode characters during markup removal (§5 step 5 keeps "the text
    inside"); before this fix `\alpha^{r}` lost its α.

Items 10-12 were found by reading the pilot outputs (Prajnya only) before any
scored paper had been scored.

### A3 (2026-11-20, during the draw; no tool had been run on any candidate)

The first draw attempt was stopped and its result discarded:

1. **Column test replaced.** A1's test (share of text runs starting right of
   the centre) called every paper two-column, because PDF text runs split at
   each font change. The test is now: on pages 2-4, group text runs into
   lines by vertical position (3-point bands); a line *crosses* the centre if
   its runs reach 20 points either side of it with no gap wider than 6
   points at the centre. Two-column when fewer than 30% of lines cross. On the
   247 PDFs already fetched it gives PLOS and eLife one-column, the other
   journals two-column (one Front Genet paper one-column).
2. **Swap search capped.** A1 rule 6 now gives up on a stratum after 30
   eligible papers without the needed feature (the uncapped search would have
   walked all 115,057 Frontiers candidates looking for a one-column paper).
   Papers passed over this way are not returned to the pool.

The candidate lists, the seed and every other rule are unchanged; the draw is
re-run from the same cached listings.

### A2 (2026-11-20, during C1 smoke tests; no corpus paper had been run)

Two tools no longer match §4's description; each is run in the configuration
its maintainers provide for this hardware:

1. **Marker 2.2.0** sends its Surya OCR model to a vLLM server in Docker on
   NVIDIA cards, and its own GPU table starts at 16 GB; our card has 8 GB.
   Without a usable server it silently falls back to a CPU OCR path
   ("Indra"). Marker is therefore run with its documented **llama.cpp
   backend** (`SURYA_INFERENCE_BACKEND=llamacpp`), pointing
   `LLAMA_CPP_BINARY` at the same llama.cpp build Prajnya uses (b12035), with
   Surya's own GGUF weights (`datalab-to/surya-ocr-2.5-gguf`). Also
   `PYTHONUTF8=1`: without it Marker crashes on Windows' cp1252 codepage.
2. **MinerU 4.2.3** replaced its backends with **tiers** of one engine
   (layout pipeline + VLM). Its local default tier is `standard`
   (= engine effort `high`). MinerU is run through its engine API
   (`mineru.backend.analyze.doc_analyze`, effort `high`) by
   `benchmark/runners/run_mineru.py`, not through its document-library
   server, which carries telemetry (on by default) and a remote-parse option
   (mineru.net). One MinerU row; §4's "VLM backend" second row no longer
   exists in this version. `MINERU_TELEMETRY_DISABLED=1` is set regardless.
