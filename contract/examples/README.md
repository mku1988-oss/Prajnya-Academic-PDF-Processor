# Contract examples

Two kinds of example live here. Both follow `../prajnya_document.schema.json`.

| file | what it is |
|---|---|
| `citations_online.json` | hand-written: a reference list and linked in-text marks |
| `citations_local_file_name.json` | hand-written: a document cited by its file name |
| `frontiers_acidovorax_hrp/` | **real output**, one open-access paper processed end to end |

## `frontiers_acidovorax_hrp/`: a real document

Prajnya's output for one 12-page research article, exactly as the run wrote it.
Nothing was edited by hand. The paper is one of the 40 scored papers of the
benchmark ([`../../BENCHMARK.md`](../../BENCHMARK.md)), and this is the output
that was scored.

```
frontiers_acidovorax_hrp/
├── PMC5880930.v6.json                   the document JSON (contract 1.3)
├── figures/PMC5880930/manifest.json     all 8 figures (captions, boxes, labels)
├── figures/PMC5880930/p*_fig*.png       their 8 crops
├── tables/PMC5880930/manifest.json      the table, with its rebuilt grid
└── tables/PMC5880930/p3_tab1.png        its crop
```

The folder keeps the layout of a run directory, so every `image_path` in the
manifests is relative to this folder, and every crop is included. The PDF
itself is not: get it from the link below.

Processed by the code released as v6.7.0 (its JSON still reads `v6.6.4`, the
number before the release raised it) at citation level `online`. In it:

- **pages:** the title page read by a vision model (`two_shot_title_page`),
  3 pages from the text layer (`pymupdf`), 7 full-page vision reads
  (`glm_fullpage`), and 1 page repaired from its table's own crop
  (`glm_repair`);
- **metadata:** title, all 8 authors in order, journal, year and DOI, confirmed
  by Crossref;
- **figures and tables:** all 8 figures under their printed labels, and
  Table 1 with its grid;
- **mentions:** 12 figure mentions and 1 table mention, every one linked;
  6 mentions of supplementary figures are counted and left unlinked, as the
  contract specifies;
- **citations:** a reference list of 65 entries (Crossref, checked against the
  65 printed in the PDF), and 92 author–year citations, each linked to its
  entries; 8 of them carry `content_offset: null` (allowed by the contract: the
  citation was not found exactly once in its page's text).

**What is not perfect, left as written.** Four of the eight crops are flawed:
Figure 1's crop includes its caption, Figure 3's cuts off the right-hand labels
of its leaf photograph, Figure 6's cuts panel B in half, and Figure 7's holds
only panel C (panels A and B are missing). Figure and table *labels* and
*captions* are all right; crop quality is not yet scored by the benchmark.

### Attribution

Zhang X, Zhao M, Yan J, Yang L, Yang Y, Guan W, Walcott R, Zhao T (2018).
Involvement of hrpX and hrpG in the Virulence of *Acidovorax citrulli* Strain
Aac5, Causal Agent of Bacterial Fruit Blotch in Cucurbits. *Front. Microbiol.*
9:507. doi:[10.3389/fmicb.2018.00507](https://doi.org/10.3389/fmicb.2018.00507).
Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Original: <https://www.frontiersin.org/articles/10.3389/fmicb.2018.00507/full>,
also in PubMed Central as PMC5880930.

The text, figures and table derived from the article remain under CC BY 4.0;
the derivation is the processing described above.
