# Contract examples

Two kinds of example live here. Both follow `../prajnya_document.schema.json`.

| file | what it is |
|---|---|
| `citations_online.json` | hand-written: a reference list and linked in-text marks |
| `citations_local_file_name.json` | hand-written: a document cited by its file name |
| `frontiers_hippo_hydra/` | **real output**, one open-access paper processed end to end |

## `frontiers_hippo_hydra/`: a real document

Prajnya's output for one 19-page research article, exactly as the run wrote it.
Nothing was edited by hand.

```
frontiers_hippo_hydra/
├── fgene-12-676182.v6.json                 the document JSON
├── figures/fgene-12-676182/manifest.json   all 10 figures (captions, boxes)
├── figures/fgene-12-676182/p6_fig1.png     Figure 1A, domain architecture
├── figures/fgene-12-676182/p15_fig1.png    Figure 9, line plots
├── tables/fgene-12-676182/manifest.json    the 1 table, with its cell grid
└── tables/fgene-12-676182/p6_tab1.png      Figure 1B, read as a table
```

The folder keeps the layout of a run directory, so every `image_path` in the
manifests is relative to this folder. To keep it small, **only 3 of the 11
crops are included**; the manifests still list all of them, and the paths of the
other 8 do not resolve here. The PDF itself is not included: get it from the
link below.

Processed by Prajnya v6.6.1 at citation level `online`. In it: metadata from
the text layer, confirmed by Crossref; 9 text-layer pages, 8 full-page vision
reads, 1 repaired page and the title page; 80 references; 103 in-text marks,
all linked to a reference (19 carry `content_offset: null`, which the contract
allows, because the mark is not found exactly once in its page text).

### Attribution

Unni M, Reddy PC, Pal M, Sagi I, Galande S (2021). Identification of Components
of the Hippo Pathway in Hydra and Potential Role of YAP in Cell Division and
Differentiation. *Front. Genet.* 12:676182.
doi:[10.3389/fgene.2021.676182](https://doi.org/10.3389/fgene.2021.676182).
Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Processed by Prajnya v6.6.1 (citations level: online). Original:
<https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2021.676182/full>.

The text, figures and table derived from the article remain under CC BY 4.0;
the derivation is the processing described above.
