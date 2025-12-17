# Acknowledgments

Prajnya stands on the shoulders of giants. We are deeply grateful to the open-source community and the creators of the following technologies that make this project possible.

## Core Infrastructure

**[llama.cpp](https://github.com/ggml-org/llama.cpp)** by ggml-org
*LLM inference in C/C++ with GPU acceleration*
The high-performance inference engine that powers our local VLM deployment option. Enables privacy-first document processing without cloud dependencies.
- GitHub: https://github.com/ggml-org/llama.cpp
- License: MIT

**[NiceGUI](https://github.com/zauberzeug/nicegui)** by Zauberzeug GmbH
*Python-based UI framework for web applications*
Powers Prajnya Studio's responsive, reactive interface. Makes complex configuration accessible through an elegant GUI.
- GitHub: https://github.com/zauberzeug/nicegui
- License: MIT
- Website: https://nicegui.io

## PDF & Document Processing

**[PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF)** by Artifex Software
*Python bindings for the MuPDF library*
Our primary PDF rendering and text extraction engine. Handles Tier 1 fast extraction for simple pages, saving 90% of processing costs.
- GitHub: https://github.com/pymupdf/PyMuPDF
- License: GNU AGPL v3
- Documentation: https://pymupdf.readthedocs.io

**[pypdfium2](https://github.com/pypdfium2-team/pypdfium2)** by pypdfium2-team
*Python bindings for PDFium*
Alternative PDF rendering engine for fallback scenarios.
- GitHub: https://github.com/pypdfium2-team/pypdfium2
- License: Apache 2.0 / BSD 3-Clause

**[Pillow (PIL Fork)](https://github.com/python-pillow/Pillow)** by Alex Clark and Contributors
*The friendly PIL fork*
Image manipulation library for page rendering, cropping, and surgical VLM extraction.
- GitHub: https://github.com/python-pillow/Pillow
- License: HPND License
- Documentation: https://pillow.readthedocs.io

**[pytesseract](https://github.com/madmaze/pytesseract)** by Matthias A Lee
*Python wrapper for Google's Tesseract OCR*
Tier 5 fallback OCR engine for scanned documents and corrupted text layers.
- GitHub: https://github.com/madmaze/pytesseract
- License: Apache 2.0
- Underlying: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) by Google

## Vision Language Models & AI

**[Google Gemini API](https://ai.google.dev/)** by Google DeepMind
*Multimodal AI models (Gemini 2.0 Flash, Gemini 2.5 Pro)*
Cloud VLM provider for fast, accurate document analysis. Powers our hybrid cloud pipeline option.
- Documentation: https://ai.google.dev/
- Python SDK: [google-genai](https://pypi.org/project/google-genai/)

**[OpenAI Vision API](https://platform.openai.com/)** by OpenAI
*GPT-4o, GPT-4-Vision models*
Alternative cloud VLM provider for highest-quality reasoning on complex layouts.
- Documentation: https://platform.openai.com/docs/guides/vision
- Python SDK: [openai](https://pypi.org/project/openai/)

**[olmOCR-2-7B](https://huggingface.co/allenai/olmOCR-2-7B-1025)** by AllenAI
*Open-source OCR-specialized Vision Language Model (v2)*
Our primary tested local VLM for privacy-first deployment. 7B parameter model optimized for document understanding. **Prajnya V5 was developed and tested with this model** (Q5_K_M quantization from bartowski).
- HuggingFace (Original): https://huggingface.co/allenai/olmOCR-2-7B-1025
- HuggingFace (GGUF): https://huggingface.co/bartowski/allenai_olmOCR-2-7B-1025-GGUF
- License: Apache 2.0
- Collection: [OlmoCR Models](https://huggingface.co/collections/allenai/olmocr-66e2fa3c3be0c2fa04589ed4)

## Metadata Validation & APIs

**[Zotero](https://www.zotero.org/)** by Corporation for Digital Scholarship
*Reference management software with powerful APIs*
Enables our triple-referee metadata validation system. Cross-validates extracted DOIs, ISBNs, and author names against authoritative databases.
- Website: https://www.zotero.org
- API Documentation: https://www.zotero.org/support/dev/web_api/v3/basics
- Python Library: [pyzotero](https://github.com/urschrei/pyzotero) by Stephan Hügel

**[Crossref API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)** by Crossref
*Scholarly metadata API*
DOI resolution and metadata lookup for academic papers.
- Documentation: https://www.crossref.org/documentation/retrieve-metadata/rest-api/
- License: CC0 (metadata)

## Python Core Dependencies

**[Pydantic](https://github.com/pydantic/pydantic)** by Samuel Colvin
*Data validation using Python type hints*
Powers our structured JSON output with validation and serialization.
- GitHub: https://github.com/pydantic/pydantic
- License: MIT

**[PyYAML](https://github.com/yaml/pyyaml)** by Kirill Simonov
*YAML parser and emitter for Python*
Enables our hot-reload configuration system and modular strategy definitions.
- GitHub: https://github.com/yaml/pyyaml
- License: MIT

**[aiohttp](https://github.com/aio-libs/aiohttp)** by aio-libs
*Asynchronous HTTP client/server framework*
Powers our async pipeline for non-blocking batch processing.
- GitHub: https://github.com/aio-libs/aiohttp
- License: Apache 2.0

**[requests](https://github.com/psf/requests)** by Kenneth Reitz
*HTTP library for Python*
Synchronous API calls for Zotero and Crossref integrations.
- GitHub: https://github.com/psf/requests
- License: Apache 2.0

**[tqdm](https://github.com/tqdm/tqdm)** by Casper da Costa-Luis
*Fast, extensible progress bar*
Provides real-time progress feedback in CLI mode.
- GitHub: https://github.com/tqdm/tqdm
- License: MPL 2.0 / MIT

**[python-dotenv](https://github.com/theskumar/python-dotenv)** by Saurabh Kumar
*Reads key-value pairs from .env file*
Manages environment configuration and API keys securely.
- GitHub: https://github.com/theskumar/python-dotenv
- License: BSD 3-Clause

**[colorlog](https://github.com/borntyping/python-colorlog)** by Sam Clements
*Colored logging for Python*
Enhances debugging with color-coded log output.
- GitHub: https://github.com/borntyping/python-colorlog
- License: MIT

## Model Quantization & Distribution

**[mradermacher](https://huggingface.co/mradermacher)** on HuggingFace
*Community GGUF quantizer*
Provides high-quality quantized GGUF versions of thousands of models. Our recommended source for downloading optimized VLMs.
- HuggingFace: https://huggingface.co/mradermacher
- Model Requests: https://huggingface.co/mradermacher/model_requests
- Special Thanks: nethype GmbH for server infrastructure

**[Unsloth AI](https://huggingface.co/unsloth)** on HuggingFace
*Efficient fine-tuning and GGUF quantization*
Official Unsloth Dynamic 2.0 GGUF quantizations. High-quality alternative model source.
- HuggingFace: https://huggingface.co/unsloth
- Website: https://unsloth.ai
- GitHub: https://github.com/unslothai/unsloth

**[GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)** by Georgi Gerganov
*Quantized model format specification*
The efficient binary format that enables local VLM deployment on consumer hardware.
- Specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Part of: [ggml](https://github.com/ggerganov/ggml) library

## Development & Research Tools

**[pytest](https://github.com/pytest-dev/pytest)** by Holger Krekel, Bruno Oliveira, et al.
*Testing framework*
Our test suite framework ensuring production reliability.
- GitHub: https://github.com/pytest-dev/pytest
- License: MIT

## Research & Benchmarking

**[OmniDocBench](https://github.com/opendatalab/OmniDocBench)** by OpenDataLab
*Comprehensive document parsing benchmark (CVPR 2025)*
The benchmark we reference for competitive comparisons. Validates our accuracy claims against state-of-the-art.
- GitHub: https://github.com/opendatalab/OmniDocBench
- Paper: [OmniDocBench: Benchmarking Diverse PDF Document Parsing (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Ouyang_OmniDocBench_Benchmarking_Diverse_PDF_Document_Parsing_with_Comprehensive_Annotations_CVPR_2025_paper.pdf)

## Infrastructure

**[Shastrasaara V4](https://github.com/mku1988-oss/Shastrasaara)** by Manu Unni
*Parent research project*
The comprehensive document intelligence research initiative within which Prajnya was developed.

## Special Thanks

- **Academic Community**: For testing, feedback, and bug reports during development
- **Open-Source Contributors**: To all who have submitted issues, PRs, and feature requests
- **VLM Model Developers**: AllenAI (OlmoCR), Google DeepMind (Gemini), OpenAI (GPT-4V), Meta (LLaMA) teams
- **Document AI Researchers**: For advancing the field and making RAG pipelines possible
- **Early Adopters**: Researchers and enterprises who trusted Prajnya in its early releases

## Standing on Open Foundations

This project would not exist without the tireless work of open-source maintainers, researchers publishing their work openly, and communities sharing knowledge freely. Prajnya is our contribution back to that ecosystem.
