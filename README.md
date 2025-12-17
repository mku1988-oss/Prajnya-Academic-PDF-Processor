# Prajnya Academic PDF Processor v5.0.0

**High-quality document intelligence for RAG pipelines**

Prajnya is a production-ready academic PDF processing engine that extracts structured JSON from research papers, books, and technical documents. Designed as a plug-and-play module for RAG (Retrieval-Augmented Generation) pipelines, it serves as the high-quality document understanding core that transforms complex PDFs into clean, structured data.

## What Does It Do?

Prajnya processes academic PDFs and outputs:
- **Structured JSON for every page**: Complete page-by-page analysis with layout understanding
- **Rich metadata extraction**: Authors, DOI, journal, publication year, abstract
- **Content extraction**: Text, equations (LaTeX), tables, figures with descriptions
- **Intelligent routing**: Automatically chooses optimal processing strategy per page
- **Vision-first analysis**: Uses Vision Language Models (VLMs) to "see" and understand document layouts

## Why Prajnya for RAG?

Traditional PDF extraction tools fail on academic papers due to:
- Complex multi-column layouts
- Mathematical equations and chemical formulas
- Figures, tables, and captions scattered across pages
- Poor text ordering and reading flow

**Prajnya solves this** by combining Vision Language Models with adaptive routing, delivering clean structured output that RAG systems can reliably consume.

## Key Features

###  **Plug-and-Play RAG Integration**
- **Modular design**: Import as Python package or use via CLI/API
- **Structured JSON output**: Ready for vector databases and embeddings
- **Batch processing**: Handle hundreds of papers with async processing
- **Persistence layer**: Atomic ledger system prevents data loss

###  **Intelligent Processing**
- **Adaptive routing**: Fast heuristics for simple pages, VLM for complex layouts
- **Vision-first analysis**: 7B+ parameter VLMs understand document structure
- **Multi-tier fallback**: Graceful degradation ensures complete coverage
- **Reading order recovery**: Maintains logical text flow across columns

###  **Production Ready**
- **Async architecture**: Non-blocking processing for high throughput
- **Error recovery**: Handles malformed PDFs gracefully
- **Progress tracking**: Real-time status for long-running batches
- **Hot-reload configs**: Update processing strategies without restart

## Output Format

### Structured JSON Schema

**Page-level output:**
```json
{
  "page_number": 1,
  "page_type": "title_page",
  "content": {
    "text": "Full extracted text in reading order...",
    "equations": [
      {"latex": "E = mc^2", "bbox": [x, y, w, h]}
    ],
    "figures": [
      {"description": "Schematic of...", "bbox": [x, y, w, h]}
    ],
    "tables": [
      {"caption": "Results summary", "data": [...]}
    ]
  },
  "metadata": {
    "title": "Paper title",
    "authors": ["Author 1", "Author 2"],
    "doi": "10.xxxx/xxxxx",
    "journal": "Nature",
    "year": 2024
  },
  "confidence": {
    "overall": 0.95,
    "fields": {"title": 0.98, "authors": 0.94, ...}
  }
}
```

**Document-level output:**
```json
{
  "document_id": "unique_hash",
  "metadata": {...},
  "pages": [{page1}, {page2}, ...],
  "processing_stats": {
    "total_pages": 26,
    "vlm_pages": 4,
    "heuristic_pages": 22,
    "processing_time": 145.3
  }
}
```

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/YourUsername/Prajnya-Academic-PDF-Processor.git
cd Prajnya-Academic-PDF-Processor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create `.env` file:
```env
# VLM Provider (choose one)
OPENAI_API_KEY=sk-...           # For GPT-4V
GEMINI_API_KEY=...              # For Gemini 2.0 Flash

# Optional: Local VLM Server
LOCAL_VLM_SERVER=http://localhost:8080/v1/chat/completions

# Optional: Zotero Integration
ZOTERO_USER_ID=...
ZOTERO_API_KEY=...
```

## Usage

### CLI (Command Line Interface)

**Process single PDF:**
```bash
python run_Prajnya.py --pdf "path/to/paper.pdf"
# Output: results saved to ./output/paper.json
```

**Batch process folder:**
```bash
python run_Prajnya.py --folder "path/to/papers/" --recursive
# Output: ./output/{paper1}.json, ./output/{paper2}.json, ...
```

**Review extracted data:**
```bash
python run_Prajnya.py --review
# Opens interactive ledger browser
```

### Python Module

```python
from src.processing.document_orchestrator import DocumentOrchestrator

# Initialize orchestrator
orchestrator = DocumentOrchestrator()

# Process PDF
result = orchestrator.process_pdf("paper.pdf")

# Access structured output
print(result['metadata'])       # Document metadata
print(result['pages'][0])       # First page analysis
print(result['pages'][5]['content']['equations'])  # Page 5 equations
```

### GUI (Prajnya Studio)

Visual interface for configuration and monitoring:
```bash
python -m prajnya_studio.studio_app
```
Open `http://localhost:8080` in browser.

**Features:**
- Strategy Lab: Configure processing strategies visually
- Batch Commander: Monitor batch processing with real-time logs
- Ledger Master: Browse and export extracted data

## RAG Pipeline Integration

### Example: Vector Database Ingestion

```python
from src.processing.document_orchestrator import DocumentOrchestrator
import chromadb

# Process papers
orchestrator = DocumentOrchestrator()
papers = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]

# Initialize vector DB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("research_papers")

# Ingest structured output
for paper_path in papers:
    result = orchestrator.process_pdf(paper_path)

    # Each page becomes a document
    for page in result['pages']:
        collection.add(
            documents=[page['content']['text']],
            metadatas=[{
                'source': paper_path,
                'page': page['page_number'],
                'doi': result['metadata'].get('doi'),
                'title': result['metadata'].get('title')
            }],
            ids=[f"{result['document_id']}_page_{page['page_number']}"]
        )
```

### Example: LangChain Integration

```python
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.processing.document_orchestrator import DocumentOrchestrator

# Process with Prajnya
orchestrator = DocumentOrchestrator()
result = orchestrator.process_pdf("paper.pdf")

# Load into LangChain
pages_text = [page['content']['text'] for page in result['pages']]
metadata = result['metadata']

# Split and embed
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = text_splitter.create_documents(
    pages_text,
    metadatas=[metadata] * len(pages_text)
)
```

## Configuration

### Processing Strategies

Edit `src/processing/strategies/config/research_article_strategy.yaml`:

```yaml
# Quick Settings
preset: "balanced"  # fast | balanced | accurate

# VLM Configuration
vlm_profile: "olmocr-7b"  # Local model
# vlm_profile: "gemini-2-5-flash"  # Cloud API

# Concurrency
concurrency:
  enabled: true
  max_concurrent: 1  # Based on VRAM

# Adaptive Routing
routing:
  complexity_threshold: 0.7
  enable_heuristic_fallback: true
```

### VLM Profiles

Choose from pre-configured profiles:
- `olmocr-7b` - Local, privacy-first (requires llama.cpp server)
- `olmocr-2-7b` - Faster local model
- `gemini-2-5-flash` - Google Cloud API (3x faster, costs $)
- `gemini-2-5-pro` - Highest quality Cloud API

## Architecture

```
Input PDF
    ↓
Document Classifier
    ↓
Strategy Router (Research Article | Book | Chapter)
    ↓
Page-by-Page Processing
    ├─ Tier 1: Fast Heuristics (simple pages)
    └─ Tier 2: VLM Analysis (complex pages)
         ├─ Layout detection
         ├─ Text extraction with reading order
         ├─ Equation → LaTeX conversion
         ├─ Figure/table description
         └─ Metadata validation
    ↓
Structured JSON Output + Ledger Storage
```

## Performance

**Benchmark** (26-page research article):
- **Fast preset**: ~8 minutes (mostly heuristics)
- **Balanced preset**: ~20 minutes (selective VLM)
- **Accurate preset**: ~45 minutes (full VLM analysis)

**Async processing** (8GB VRAM, 1 concurrent VLM):
- Throughput: ~3-4 papers/hour (balanced preset)
- With 16GB VRAM: ~6-8 papers/hour (2 concurrent)

## Requirements

**Minimum:**
- Python 3.10+
- 8GB RAM
- 10GB disk space

**Recommended for local VLM:**
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+
- llama.cpp server (included in docs)

**Cloud API only:**
- No GPU required
- Internet connection for API calls

## Use Cases

✅ **RAG Pipeline Core**: High-quality document understanding for LLM context
✅ **Literature Review**: Batch process hundreds of papers for systematic review
✅ **Knowledge Graphs**: Extract structured data for graph construction
✅ **Citation Analysis**: Build citation networks from references
✅ **Dataset Creation**: Generate training data for ML models
✅ **Metadata Enrichment**: Enhance library catalogs with deep extraction

## Limitations

- Optimized for academic papers (journals, preprints, technical reports)
- Scanned PDFs require OCR preprocessing
- Handwritten annotations not supported
- Non-Latin scripts have reduced accuracy

## License

MIT License - Free for commercial and academic use

## Citation

If you use Prajnya in your research, please cite:
```bibtex
@software{prajnya2024,
  title={Prajnya: Academic PDF Processor for RAG Pipelines},
  author={Manu Unni},
  year={2026},
  url={https://github.com/mku1988-oss/Prajnya-Academic-PDF-Processor/}
}
```

## Support & Contributing

- **Issues**: [GitHub Issues](https://github.com/YourUsername/Prajnya-Academic-PDF-Processor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YourUsername/Prajnya-Academic-PDF-Processor/discussions)
- **Documentation**: [Full Documentation](https://github.com/YourUsername/Prajnya-Academic-PDF-Processor/wiki)

---

**Built for researchers, by researchers.** 🔬📚
