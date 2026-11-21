# Benchmark tools: versions and settings (protocol §4)

Committed before the pilot (C4). Every setting that is not the tool's default
is listed with its reason. Each tool runs in its own environment; none is a
Prajnya dependency.

**Machine:** Windows 11 Home 10.0.26300; NVIDIA GeForce RTX 4070 Laptop GPU,
8 GB (driver 617.42); 31.7 GB RAM; Docker Desktop 28.0.4 with WSL2 capped at
10 GB (`.wslconfig`).

| tool | version | licence | environment | settings (non-default in bold) |
|---|---|---|---|---|
| Prajnya | v6.6.4 (tag) | Apache-2.0 | its own conda env (Python 3.10.16); llama-server b12035, `--models-max 1` | released YAML (`citations.level: local`); `online` row: **`--citations online`** |
| GROBID | 0.9.1 (`grobid/grobid:0.9.1-full`, image `35c83afedde0`) | Apache-2.0 | Docker, CPU, container `grobid_bench`, port 8070 | `/api/processFulltextDocument`; offline row: **`consolidateHeader=0`, `consolidateCitations=0`**; consolidated row: **`=1`, `=1`**; **`teiCoordinates=figure`** (adds figure boxes, changes no text) |
| Docling | 2.151.1 | MIT | conda env `bench_docling` (Python 3.11, torch cu128) | default `DocumentConverter` |
| Marker | marker-pdf 2.2.0 (surya-ocr 0.23.0) | Apache-2.0 (code; model weights: see their licence) | conda env `bench_marker` (Python 3.11, torch cu128) | `marker_single --output_format json`, no LLM option; **`SURYA_INFERENCE_BACKEND=llamacpp`**, **`LLAMA_CPP_BINARY=<llama-server>`** (b12035), **`PYTHONUTF8=1`** (amendment A2) |
| MinerU | 4.2.3 (bundled `mineru_llama_cpp` 0.2.3) | MinerU Open Source License | conda env `bench_mineru` (Python 3.11, torch cu128) | engine API via `benchmark/runners/run_mineru.py`, effort `high` = default local tier `standard`; **`MINERU_TELEMETRY_DISABLED=1`**, **`PYTHONUTF8=1`** (amendment A2) |

Docling's models are downloaded to `HF_HOME` (the Hugging Face cache).

## GROBID model set (read from the image's `grobid.yaml`)

DeLFT BiLSTM-CRF with layout features (no ELMo, no transformer) for `header`,
`reference-segmenter`, `citation`, `affiliation-address`,
`funding-acknowledgement`; Wapiti CRF for `segmentation`, `fulltext`,
`figure`, `table`, `name-*`, `date`.

## Smoke tests (C1)

| tool | PDF (not in corpus) | result |
|---|---|---|
| GROBID | `fgene-12-676182.pdf` (19 pp.) | HTTP 200; 3.3 s warm; title, 6 authors, DOI, date; journal missing; 9 figures; 82 references; 107/112 citation refs and 70/70 figure refs linked. Two runs differed in the funder block. |
| Docling | same | 2 runs: 112 s (first; includes 44 s model download), 20 s; peak VRAM 2.0 GB. 13 pictures, 1 table, 9 captions. |
| Marker | same | Default route fell back to CPU "Indra" OCR (no Docker on PATH; vLLM needs ≥ 16 GB). With llama.cpp backend: 288 s (first; downloads), 141 s; peak VRAM 3.2 GB. 10 figure/picture blocks, 6 captions, 0 tables. |
| MinerU | same | 2 runs: 20 s, 20 s (after a first run that downloaded models, 207 s); peak VRAM 2.8 GB; VLM via bundled llama.cpp on GPU. 12 images, 2 charts, 1 table. |

Counts are what each tool emitted, not scores. Scoring needs the JATS key.
