# Results tables: run 2, scored papers

Each cell: mean over papers (number of papers scored). `n/s`: not supported by the tool. Failed papers are scored as empty output.

| tool | papers | failed | network |
|---|---|---|---|
| Raw text layer (baseline) | 40 | 0 | none |
| Prajnya (local) | 40 | 0 | none |
| Prajnya (online) | 40 | 0 | DOI/title to Crossref, OpenAlex |
| GROBID | 40 | 0 | none |
| GROBID (consolidated) | 40 | 0 | extracted fields to Crossref |
| Docling | 40 | 0 | none |
| Marker | 40 | 0 | none |
| MinerU | 40 | 0 | none |

## Metadata (§6.1)

| metric | Raw text layer (baseline) | Prajnya (local) | Prajnya (online) | GROBID | GROBID (consolidated) | Docling | Marker | MinerU |
|---|---|---|---|---|---|---|---|---|
| title_exact | n/s | 0.975 (40) | 1.000 (40) | 0.825 (40) | 0.975 (40) | n/s | n/s | n/s |
| authors_f1 | n/s | 0.945 (40) | 0.952 (40) | 0.943 (40) | 1.000 (40) | n/s | n/s | n/s |
| authors_order | n/s | 0.650 (40) | 0.650 (40) | 0.450 (40) | 1.000 (40) | n/s | n/s | n/s |
| doi | n/s | 0.975 (40) | 0.975 (40) | 0.975 (40) | 0.975 (40) | n/s | n/s | n/s |
| year | n/s | 1.000 (40) | 1.000 (40) | 0.800 (40) | 1.000 (40) | n/s | n/s | n/s |
| journal | n/s | 1.000 (40) | 1.000 (40) | 0.000 (40) | 0.975 (40) | n/s | n/s | n/s |

## Abstract and body text (§6.2)

| metric | Raw text layer (baseline) | Prajnya (local) | Prajnya (online) | GROBID | GROBID (consolidated) | Docling | Marker | MinerU |
|---|---|---|---|---|---|---|---|---|
| abstract_similarity | n/s | n/s | n/s | 0.701 (39) | 0.701 (39) | n/s | n/s | n/s |
| body_coverage | 0.884 (40) | 0.960 (40) | 0.960 (40) | 0.891 (40) | 0.891 (40) | 0.956 (40) | 0.919 (40) | 0.949 (40) |
| body_similarity | 0.473 (40) | 0.727 (40) | 0.727 (40) | 0.846 (40) | 0.846 (40) | 0.542 (40) | 0.715 (40) | 0.734 (40) |
| reading_order_tau | 1.000 (40) | 1.000 (40) | 1.000 (40) | 1.000 (40) | 1.000 (40) | 1.000 (40) | 1.000 (40) | 1.000 (40) |

## Figures (§6.3)

| metric | Raw text layer (baseline) | Prajnya (local) | Prajnya (online) | GROBID | GROBID (consolidated) | Docling | Marker | MinerU |
|---|---|---|---|---|---|---|---|---|
| figures_recall | n/s | 0.935 (40) | 0.935 (40) | 0.876 (40) | 0.876 (40) | 0.746 (40) | 0.684 (40) | 0.635 (40) |
| figures_recall_with_caption_match | n/s | 0.935 (40) | 0.935 (40) | 0.914 (40) | 0.914 (40) | 0.768 (40) | 0.689 (40) | 0.806 (40) |
| figures_precision | n/s | 1.000 (40) | 1.000 (40) | 1.000 (39) | 1.000 (39) | 1.000 (40) | 1.000 (40) | 1.000 (38) |
| figure_caption_similarity | n/s | 0.785 (40) | 0.785 (40) | 0.831 (39) | 0.831 (39) | 0.672 (40) | 0.646 (40) | 0.762 (38) |
| figures_kept_together | n/s | 0.935 (40) | 0.935 (40) | 0.801 (40) | 0.801 (40) | 0.746 (40) | 0.684 (40) | 0.635 (40) |
| figures_kept_together_with_caption_match | n/s | 0.935 (40) | 0.935 (40) | 0.839 (40) | 0.839 (40) | 0.768 (40) | 0.689 (40) | 0.806 (40) |

## Tables (§6.4)

| metric | Raw text layer (baseline) | Prajnya (local) | Prajnya (online) | GROBID | GROBID (consolidated) | Docling | Marker | MinerU |
|---|---|---|---|---|---|---|---|---|
| tables_recall | n/s | 0.892 (17) | 0.892 (17) | 0.706 (17) | 0.706 (17) | 0.696 (17) | 0.892 (17) | 0.814 (17) |
| tables_precision | n/s | 1.000 (16) | 1.000 (16) | 1.000 (13) | 1.000 (13) | 1.000 (13) | 1.000 (16) | 1.000 (15) |
| teds | n/s | 0.942 (16) | 0.942 (16) | 0.606 (13) | 0.606 (13) | 0.960 (13) | 0.883 (16) | 0.947 (15) |
| teds_structure | n/s | 0.974 (16) | 0.974 (16) | 0.648 (13) | 0.648 (13) | 0.964 (13) | 0.921 (16) | 0.958 (15) |
| tables_unlabelled | n/s | 0.941 (17) | 0.941 (17) | 0.000 (17) | 0.000 (17) | 1.353 (17) | 1.353 (17) | 1.588 (17) |

## References (§6.5)

| metric | Raw text layer (baseline) | Prajnya (local) | Prajnya (online) | GROBID | GROBID (consolidated) | Docling | Marker | MinerU |
|---|---|---|---|---|---|---|---|---|
| references_recall | n/s | n/s | 0.997 (40) | 0.991 (40) | 0.972 (40) | n/s | n/s | 0.999 (40) |
| references_recall_frozen_rule | n/s | n/s | 0.948 (40) | 0.946 (40) | 0.941 (40) | n/s | n/s | 0.656 (40) |
| references_precision | n/s | n/s | 0.994 (40) | 0.983 (40) | 0.965 (40) | n/s | n/s | 0.988 (40) |
| ref_title | n/s | n/s | 0.963 (39) | 0.988 (39) | 0.977 (39) | n/s | n/s | n/s |
| ref_year | n/s | n/s | 0.998 (39) | 0.982 (39) | 0.982 (39) | n/s | n/s | n/s |
| ref_doi | n/s | n/s | 0.915 (40) | 0.663 (40) | 0.935 (40) | n/s | n/s | n/s |
| ref_first_author | n/s | n/s | 0.998 (40) | 0.985 (40) | 0.988 (40) | n/s | n/s | n/s |

## In-text links (§6.6, §6.7)

| metric | Raw text layer (baseline) | Prajnya (local) | Prajnya (online) | GROBID | GROBID (consolidated) | Docling | Marker | MinerU |
|---|---|---|---|---|---|---|---|---|
| citation_precision | n/s | n/s | 0.939 (36) | 0.964 (40) | 0.948 (40) | n/s | n/s | n/s |
| citation_recall | n/s | n/s | 0.829 (40) | 0.894 (40) | 0.880 (40) | n/s | n/s | n/s |
| figure_mention_precision | n/s | 0.896 (40) | 0.896 (40) | 0.728 (39) | 0.728 (39) | n/s | n/s | n/s |
| figure_mention_recall | n/s | 0.919 (40) | 0.919 (40) | 0.788 (40) | 0.788 (40) | n/s | n/s | n/s |
| table_mention_precision | n/s | 0.896 (16) | 0.896 (16) | 0.643 (12) | 0.643 (12) | n/s | n/s | n/s |
| table_mention_recall | n/s | 0.826 (17) | 0.826 (17) | 0.596 (17) | 0.596 (17) | n/s | n/s | n/s |

## Cost (§6.9; reported, not ranked)

| tool | papers | pages | seconds per page | median seconds per paper | peak VRAM (MiB) | peak system RAM in use (GB) |
|---|---|---|---|---|---|---|
| Prajnya (local) | 40 | 761 | 11.08 | 174.8 | 7934 | 19.4 |
| GROBID | 40 | 761 | 0.17 | 3.1 | 26 | 19.6 |
| GROBID (consolidated) | 40 | 761 | 3.15 | 59.6 | 26 | 20.2 |
| Docling | 40 | 761 | 1.06 | 20.1 | 2954 | 17.5 |
| Marker | 40 | 761 | 6.28 | 116.5 | 3250 | 19.4 |
| MinerU | 40 | 761 | 1.22 | 21.0 | 3313 | 16.3 |

## By publisher (main metrics)

**body_coverage**

| publisher | Raw text layer (baseline) | Prajnya (local) | Prajnya (online) | GROBID | GROBID (consolidated) | Docling | Marker | MinerU |
|---|---|---|---|---|---|---|---|---|
| CellPress | 0.919 (8) | 0.954 (8) | 0.954 (8) | 0.833 (8) | 0.833 (8) | 0.937 (8) | 0.884 (8) | 0.960 (8) |
| Frontiers | 0.879 (8) | 0.988 (8) | 0.988 (8) | 0.886 (8) | 0.886 (8) | 0.990 (8) | 0.926 (8) | 1.000 (8) |
| NatureComms | 0.885 (8) | 0.951 (8) | 0.951 (8) | 0.831 (8) | 0.831 (8) | 0.939 (8) | 0.899 (8) | 0.919 (8) |
| PLOS | 0.879 (8) | 0.973 (8) | 0.973 (8) | 0.981 (8) | 0.981 (8) | 0.980 (8) | 0.962 (8) | 0.954 (8) |
| eLife | 0.859 (8) | 0.933 (8) | 0.933 (8) | 0.925 (8) | 0.925 (8) | 0.933 (8) | 0.924 (8) | 0.913 (8) |

**figures_recall**

| publisher | Raw text layer (baseline) | Prajnya (local) | Prajnya (online) | GROBID | GROBID (consolidated) | Docling | Marker | MinerU |
|---|---|---|---|---|---|---|---|---|
| CellPress | n/s | 0.843 (8) | 0.843 (8) | 0.868 (8) | 0.868 (8) | 0.579 (8) | 0.770 (8) | 0.496 (8) |
| Frontiers | n/s | 0.895 (8) | 0.895 (8) | 0.799 (8) | 0.799 (8) | 0.692 (8) | 0.603 (8) | 0.492 (8) |
| NatureComms | n/s | 0.934 (8) | 0.934 (8) | 0.882 (8) | 0.882 (8) | 0.735 (8) | 0.445 (8) | 0.672 (8) |
| PLOS | n/s | 1.000 (8) | 1.000 (8) | 0.982 (8) | 0.982 (8) | 0.847 (8) | 0.893 (8) | 0.618 (8) |
| eLife | n/s | 1.000 (8) | 1.000 (8) | 0.848 (8) | 0.848 (8) | 0.877 (8) | 0.708 (8) | 0.900 (8) |

**teds**

| publisher | Raw text layer (baseline) | Prajnya (local) | Prajnya (online) | GROBID | GROBID (consolidated) | Docling | Marker | MinerU |
|---|---|---|---|---|---|---|---|---|
| CellPress | n/s | 0.983 (1) | 0.983 (1) | 0.926 (1) | 0.926 (1) | n/s | 1.000 (1) | 0.983 (1) |
| Frontiers | n/s | 0.939 (6) | 0.939 (6) | 0.692 (7) | 0.692 (7) | 0.979 (6) | 0.923 (7) | 0.943 (7) |
| NatureComms | n/s | 0.868 (3) | 0.868 (3) | 0.590 (2) | 0.590 (2) | 0.945 (3) | 0.782 (3) | 0.824 (2) |
| PLOS | n/s | 0.978 (3) | 0.978 (3) | 0.354 (2) | 0.354 (2) | 0.946 (3) | 0.953 (3) | 0.995 (3) |
| eLife | n/s | 0.973 (3) | 0.973 (3) | 0.215 (1) | 0.215 (1) | 0.936 (1) | 0.732 (2) | 0.999 (2) |

**references_recall**

| publisher | Raw text layer (baseline) | Prajnya (local) | Prajnya (online) | GROBID | GROBID (consolidated) | Docling | Marker | MinerU |
|---|---|---|---|---|---|---|---|---|
| CellPress | n/s | n/s | 1.000 (8) | 0.988 (8) | 0.977 (8) | n/s | n/s | 0.997 (8) |
| Frontiers | n/s | n/s | 0.996 (8) | 0.981 (8) | 0.974 (8) | n/s | n/s | 1.000 (8) |
| NatureComms | n/s | n/s | 0.988 (8) | 0.986 (8) | 0.941 (8) | n/s | n/s | 1.000 (8) |
| PLOS | n/s | n/s | 1.000 (8) | 1.000 (8) | 0.989 (8) | n/s | n/s | 1.000 (8) |
| eLife | n/s | n/s | 0.999 (8) | 0.998 (8) | 0.982 (8) | n/s | n/s | 0.998 (8) |

**citation_recall**

| publisher | Raw text layer (baseline) | Prajnya (local) | Prajnya (online) | GROBID | GROBID (consolidated) | Docling | Marker | MinerU |
|---|---|---|---|---|---|---|---|---|
| CellPress | n/s | n/s | 0.834 (8) | 0.907 (8) | 0.898 (8) | n/s | n/s | n/s |
| Frontiers | n/s | n/s | 0.591 (8) | 0.904 (8) | 0.899 (8) | n/s | n/s | n/s |
| NatureComms | n/s | n/s | 0.829 (8) | 0.799 (8) | 0.766 (8) | n/s | n/s | n/s |
| PLOS | n/s | n/s | 0.970 (8) | 0.988 (8) | 0.978 (8) | n/s | n/s | n/s |
| eLife | n/s | n/s | 0.919 (8) | 0.873 (8) | 0.859 (8) | n/s | n/s | n/s |


## Determinism (§6.8)

| tool | papers in both runs | identical | fields that changed |
|---|---|---|---|
| Raw text layer (baseline) | 40 | 40 |  |
| Prajnya (local) | 40 | 40 |  |
| Prajnya (online) | 40 | 40 |  |
| GROBID | 40 | 40 |  |
| GROBID (consolidated) | 40 | 22 | references (18) |
| Docling | 40 | 40 |  |
| Marker | 40 | 21 | body_paragraphs (15), figures (7), tables (3), unlabelled_figure_objects (2) |
| MinerU | 40 | 37 | body_paragraphs (3), figures (1) |
