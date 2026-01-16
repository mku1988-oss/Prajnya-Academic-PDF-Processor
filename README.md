# Prajnya: The Reference Architecture for Academic RAG

> **"Don't compromise between Speed, Quality, and Privacy. Prajnya is the Intelligent Broker that negotiates all three on a per-page basis."**

![Status](https://img.shields.io/badge/Status-Production%20Ready-green?style=for-the-badge) ![Version](https://img.shields.io/badge/Version-5.0.0%20Gold-blue?style=for-the-badge) ![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge) ![Tests](https://img.shields.io/badge/Tests-Passing-green?style=for-the-badge)

---

## What is Prajnya?

Prajnya is a **Modular Document Intelligence Engine** engineered specifically for high-precision **RAG (Retrieval-Augmented Generation)** pipelines. 

Unlike generic PDF parsers that output unstructured text, Prajnya reconstructs documents into rigorous **Structured JSON**, preserving layout hierarchies, tables, and mathematical equations. It is designed to solve the "Garbage In, Garbage Out" problem in LLM applications by ensuring that the data feeding your vector database is verified, structured, and semantically rich.

### Core Capabilities
*   **Structured Extraction**: Converts PDFs into clean JSON with dedicated fields for Text, Metadata, Tables, and Figures.
*   **Layout Understanding**: Uses Vision Models to "see" the page, accurately parsing multi-column academic layouts.
*   **Hardware Aware**: Automatically detects your GPU (CUDA) and optimizes concurrency for maximum throughput.
*   **Resilient**: Built for large-scale batch processing (10,000+ files) with crash recovery and atomic state management.

---

## Try It Now: 30-Second Demo

**Single command to process your first paper:**
```bash
python run_Prajnya.py --pdf "path/to/paper.pdf" --preset fast
# Output: results.json in 2-5 minutes (depending on pages)
```

**Output Preview:**
```json
{
  "metadata": {
    "title": "Deep Learning for Climate Modeling",
    "authors": ["Smith, J.", "Zhang, L."],
    "doi": "10.1038/nature12345",
    "confidence": {"title": 0.98, "authors": 0.92}
  },
  "pages": [
    {"page_number": 1, "content": {...}, "equations": [...], "tables": [...]}
  ]
}
```

**Want to see it in action?** Launch the GUI:
```bash
python -m prajnya_studio.studio_app
# Open http://localhost:8080
```

---

## The Binary Trap: Why Existing Tools Fail

Most document processing tools force you into a bad tradeoff:
*   **Fast & Dumb** (PyMuPDF, Tesseract): $0/page, but fails on multi-column layouts, tables, and math.
*   **Smart & Expensive** (LlamaParse, GPT-4o): $0.05/page, excellent quality, but **bankrupts you at scale** ($500 for 10k pages).
*   **Locked In** (SaaS APIs): Your data leaves your server, violating privacy laws (GDPR/HIPAA).

## The Prajnya Solution: 5-Tier Adaptive Arbitrage

Prajnya doesn't just "read" PDFs. It acts as a **Smart Router**, inspecting every page first to assign the **cheapest capable agent**.

| Tier | Agent Type | Speed | Cost | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Native Extractor** (PyMuPDF) | ~0.5s | $0 | High-density text, simple layouts. |
| **Tier 2** | **Surgical VLM** (Crop & See) | ~5s | Low | Complex figures/charts (extracts region only). |
| **Tier 3** | **Single-Shot VLM** | ~15s | Med | Full-page tables, moderate complexity. |
| **Tier 4** | **Chain-of-Thought** (See-Think) | ~45s | High | **Title Pages**, complex layouts. Deep reasoning. |
| **Tier 5** | **Fallback** (OCR) | ~3s | $0 | Scanned files, corrupted text layers. |

**Result**: You process 10,000 mixed pages for **$30** (using Hybrid Cloud) or **$0** (using Local GPU), instead of **$500+** with pure VLM wrappers.

### How It Works: Per-Page Intelligence

```
┌──────────────────────────────────────────────────────┐
│               INPUT: research_paper.pdf              │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
           ┌──────────────────┐
           │  Page Classifier │ ◄── "Is this simple or complex?"
           │   (0.5s/page)    │
           └────────┬─────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌────────┐ ┌─────────┐ ┌──────────┐
   │ Tier 1 │ │ Tier 2  │ │ Tier 4   │
   │ PyMuPDF│ │ Surgical│ │ Full VLM │
   │  (85%) │ │   (10%) │ │   (5%)   │
   └────────┘ └─────────┘ └──────────┘
        │           │           │
        └───────────┼───────────┘
                    ▼
         ┌─────────────────────┐
         │  Metadata Validator │ ◄── Triple-Referee Check
         │  (Zotero + VLM + OCR)│
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │ Structured JSON Out │
         └─────────────────────┘
```

**Result**: 85% of pages use fast Tier 1 ($0). Only 5% use expensive VLM. **90% cost savings** vs pure-VLM tools.

---

## Key Differentiators

### 1. "Trust, But Verify" (The Golden Record)
Generative AI hallucinates. Prajnya **validates**.
We implement a **Triple-Referee System** for critical metadata (Title, Authors, DOI):
1.  **VLM Referee**: "I see 'Srnth' as the author."
2.  **Text Referee**: "Raw OCR shows 'Smith'."
3.  **Authority Referee**: "Zotero API confirms DOI matches 'Smith, J. (2024)'."
> **Result**: Prajnya auto-corrects the hallucination. Bad metadata = Garbage RAG. Prajnya stops it at the source.

### 2. Sovereign Intelligence (Bring Your Own Brain)
We are **Model Agnostic**. You are not locked into a vendor.
*   **Privacy First**: Run 100% offline with `olmocr-7b` or `llama-3-vision`.
*   **Speed First**: Switch to `gemini-3-flash` (Thinking-enabled). 
*   **Reasoning First**: Switch to `gpt-4o` or `claude-3-5-sonnet`.

**Configuration is just 1 line of YAML:**
```yaml
# src/processing/strategies/config/research_article_strategy.yaml
profile_registry:
  default: "gemini-3-flash"  # <--- High-velocity "Thinking" VLM
```

For detailed instructions on running **100% Offline** (Self-Hosted), see our [Advanced Setup Guide](./Advanced_Setup.md).

### 3. Surgical Precision (Visual Segmentation)
Instead of sending a full 2000x3000px page to an LLM (expensive tokens, low resolution), Prajnya detects a chart, **crops specifically that 500x500 chart**, and sends a high-res patch.

---

## Battle-Tested for Production

Unlike academic demos (Marker, Nougat), Prajnya is **production-grade** infrastructure designed for real-world deployment.

### Atomic State Management
*   **Crash-Proof Ledger**: `.csv.tmp` → `.csv` atomic moves. Power failure at file #7,382? **Zero data loss.**
*   **SHA-256 Deduplication**: Calculates hash of every PDF. Even if filenames change, Prajnya knows "I've already processed this paper" and **skips instantly**.
*   **Stateful Resume**: Resume *exactly* where you crashed in 10,000-file batches.

### Real-World Battle Testing
*   ✅ Handles 10,000+ file batches without supervision
*   ✅ Zero data loss on crashes (extensively tested)
*   ✅ Automatic error recovery and retry logic

---

## Zero to Hero: 3 Ways to Use

### Level 1: "The Researcher" (GUI)
**Tool**: **Prajnya Studio** (`prajnya_studio`)
*   **For**: Users who want "Drag & Drop".
*   **Run**: `python -m prajnya_studio.studio_app`
*   **Perfect For**: Researchers with 100-1000 papers who need a visual interface.

### Level 2: "The Automation Engineer" (CLI)
**Tool**: **Headless Mode** (`run_Prajnya.py`)
*   **For**: Cron jobs, Docker containers, CI/CD pipelines.
*   **Run**: `python run_Prajnya.py --folder ./data/papers --recursive --headless`
*   **Perfect For**: DevOps teams integrating Prajnya into data pipelines.

### Level 3: "The Architect" (Python API)
**Tool**: **Modular Library** (`DocumentOrchestrator`)
*   **For**: Building custom RAG apps.
*   **Code**:
    ```python
    from src.processing.document_orchestrator import DocumentOrchestrator
    orchestrator = DocumentOrchestrator(config="accurate")
    result = orchestrator.process_pdf("nature_paper.pdf")
    ```
*   **Perfect For**: ML engineers building production RAG systems.

---

## Competitive Landscape

| Feature | **Prajnya V5** | LlamaParse | Unstructured | MinerU | Marker |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logic** | **5-Tier Adaptive** | Agentic VLM | Partitioning | Deep Learning | End-to-End DL |
| **Cost (10k pages)** | **$30-$50** (Hybrid) | ~$300+ | Variable | Free (GPU req.) | Free (GPU req.) |
| **Privacy** | **100% Local Capable** | Cloud Only | Mixed | Local | Local |
| **Metadata Verification** | **✅ Triple-Referee** | ❌ No | ❌ No | ❌ No | ❌ No |
| **Production Features** | **✅ Ledger + Resume** | ❌ No | ❌ No | ❌ No | ❌ No |

> **"Why not MinerU?"**
> MinerU is faster (0.21s/page), but lacks **verification**. Prajnya prioritizes **Truth > Speed** for RAG reliability.
>
> **"Why not LlamaParse?"**
> LlamaParse is expensive ($300+/10k pages). Prajnya offers **90% cost savings** via adaptive routing.

---

## Installation & Setup

### Requirements
*   **Python**: 3.10+
*   **RAM**: 8GB minimum (16GB recommended for Local VLMs)
*   **VRAM** (Optional): 8GB+ for Local VLMs

### Quick Start

1.  **Clone & Install**
    ```bash
    git clone https://github.com/mku1988-oss/Prajnya-Academic-PDF-Processor.git
    cd Prajnya-Academic-PDF-Processor
    pip install -r requirements.txt
    ```

2.  **Configure Brain**
    Create `.env` file:
    ```env
    # Option 1: Google Gemini (Recommended)
    GEMINI_API_KEY=AIzaSy...
    
    # Option 2: OpenAI-Compatible Cloud (OpenRouter / OpenAI)
    OPENROUTER_API_KEY=sk-or-...
    OPENAI_API_KEY=sk-...

    # Option 3: Local VLM (Privacy-first)
    LOCAL_VLM_SERVER=http://localhost:8080/v1/chat/completions
    ```

3.  **Run Studio**
    ```bash
    python -m prajnya_studio.studio_app
    ```
    Open `http://localhost:8080`.

### Configuration Presets

Prajnya comes with 3 built-in presets:

| Preset | Speed | Cost | Accuracy | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Fast** | ⚡⚡⚡ | $ | ⭐⭐⭐ | Quick batch processing |
| **Balanced** | ⚡⚡ | $$ | ⭐⭐⭐⭐ | General-purpose (default) |
| **Accurate** | ⚡ | $$$ | ⭐⭐⭐⭐⭐ | Critical metadata extraction |

---

## Advanced Usage

For deeper technical configuration, please consult our specialized guides:

*   📖 **[Advanced Setup Guide](./Advanced_Setup.md)**: Detailed instructions for **Local VLM (llama.cpp) deployment**, **Zotero Integration**, and **Performance Tuning**.
*   ❓ **Troubleshooting**: Common solutions for CUDA, API, and Docker issues are in the [Advanced Setup Guide](./Advanced_Setup.md#3-troubleshooting).

---

## Roadmap: Phase 6 (Q1 2026)

*   **Cognitive Router**: Integration of `Dolphin-V2` (3B) as Tier 1.5 layout engine (6x speedup).
*   **Asset Ledger**: Decoupled storage for extracted Images/Charts with semantic captions.
*   **Multimodal RAG**: Native vector DB ingestion for "Search by Chart".

---

## Citation

If you use Prajnya in your research, please cite:

```bibtex
@software{prajnya2026,
  title={Prajnya: Reference Architecture for Academic RAG Pipelines},
  author={Manu Unni},
  year={2026},
  version={5.0.0},
  url={https://github.com/mku1988-oss/Prajnya-Academic-PDF-Processor}
}
```

---

## Acknowledgments

Prajnya stands on the shoulders of giants. We deeply appreciate the open-source community.

*   **Core**: [llama.cpp](https://github.com/ggml-org/llama.cpp), [NiceGUI](https://github.com/zauberzeug/nicegui)
*   **Processing**: [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [Pillow](https://github.com/python-pillow/Pillow)
*   **AI**: [AllenAI (olmOCR)](https://huggingface.co/allenai/olmOCR-2-7B-1025), [Google Gemini](https://ai.google.dev/), [OpenAI](https://openai.com/)

See the full [Acknowledgments](./Acknowledgements.md) file for comprehensive credits.

---

**Built for researchers who need truth, enterprises that require privacy, and startups that demand cost control.**

> *"Prajnya: Where adaptive intelligence meets verified knowledge."*

*Last Updated: December 18, 2024*
