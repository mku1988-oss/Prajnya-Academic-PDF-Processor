# Advanced Setup & Configuration Guide

> **This guide covers advanced deployment scenarios including OpenRouter Cloud API, Local VLM (Privacy-First), Zotero Integration (Triple-Referee Verification), and Performance Tuning.**

---

## 1. OpenRouter Cloud API (NEW in v5.1.0)

**OpenRouter** provides access to 100+ AI models through a unified API, including FREE tier models perfect for document processing.

### Why OpenRouter?

- **FREE Models**: NVIDIA Nemotron 12B VL, Meta Llama 3.2 Vision - zero cost
- **100+ Models**: Access GPT-4o, Claude, Gemini, and open-source models via single API
- **No Vendor Lock-in**: Switch models with 1 line of YAML
- **Pay-as-you-go**: No subscriptions, only pay for what you use

### Step 1: Get Your API Key

1. Visit [OpenRouter.ai](https://openrouter.ai) and create an account
2. Go to **Keys** → **Create Key**
3. Copy your API key (starts with `sk-or-v1-`)

### Step 2: Configure Environment

Add to your `.env` file:

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx
```

### Step 3: Select a Model Profile

Prajnya includes pre-configured OpenRouter profiles:

| Profile | Model | Cost | Best For |
| :--- | :--- | :--- | :--- |
| `openrouter-nemotron-v2-vl.yaml` | **NVIDIA Nemotron 12B VL** | **FREE** | General document processing |
| `openrouter-llama-vision.yaml` | **Meta Llama 3.2 11B** | **FREE** | Fast extraction |

**Use in Strategy Lab (GUI)**:
1. Open Prajnya Studio
2. Go to **Strategy Lab** → **VLM Configuration**
3. Select the OpenRouter profile from dropdown

**Use via CLI**:
```bash
python run_Prajnya.py --pdf "paper.pdf" --profile "src/processing/strategies/config/vlm_profiles/openrouter-nemotron-v2-vl.yaml"
```

### Step 4: Verify Connection

Test your setup:
```bash
python -c "from src.llm.providers.universal_vlm_provider import UniversalVLMProvider; print('OpenRouter ready!')"
```

### Recommended FREE Models for Document Processing

| Model | Strengths | OpenRouter ID |
| :--- | :--- | :--- |
| **NVIDIA Nemotron 12B VL** | Vision + Reasoning, CoT capable | `nvidia/nemotron-nano-12b-v2-vl:free` |
| **Meta Llama 3.2 11B Vision** | Fast, reliable extraction | `meta-llama/llama-3.2-11b-vision-instruct:free` |

---

## 2. Local VLM Deployment (Privacy-First)

For **100% privacy** and zero API costs, deploy local Vision Language Models using llama.cpp.

### Step 1: Install llama.cpp Server

#### Option A: Download Pre-Built Binaries (Easiest)

Visit the [llama.cpp Releases page](https://github.com/ggml-org/llama.cpp/releases) and download the appropriate binary for your system:

**Windows:**
- **AVX2** (Intel Haswell 2013+): `llama-bin-win-avx2-x64.zip`
- **CUDA** (NVIDIA GPU): `llama-bin-win-cuda-cu11.7-x64.zip` or `llama-bin-win-cuda-cu12.2-x64.zip`

**Linux:**
- **CPU**: `llama-<version>-bin-linux-x64.zip` (GPU builds require building from source)

**macOS:**
- **Intel Mac**: `llama-<version>-bin-macos-x64.zip` (CPU only)
- **Apple Silicon**: `llama-<version>-bin-macos-arm64.zip` (Metal GPU support)

Extract the archive and note the location of `llama-server` (or `llama-server.exe` on Windows).

#### Option B: Build from Source with GPU Support (Best Performance)

**Prerequisites:**
- **Windows**: Visual Studio 2022, CMake 3.14+
- **Linux**: GCC/Clang, CMake 3.14+, CUDA Toolkit (for NVIDIA GPUs)
- **macOS**: Xcode Command Line Tools, CMake

**Build Steps:**

```bash
# Clone repository
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Build with CUDA support (NVIDIA GPU - Linux/Windows)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# Build with Metal support (Apple Silicon - macOS)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release

# Build CPU-only (all platforms)
cmake -B build
cmake --build build --config Release
```

After building, the `llama-server` binary will be in the `build/bin/` directory.

---

### Step 2: Download VLM Models (GGUF Format)

Prajnya requires **Vision Language Models** in GGUF format. Choose quantization level based on your VRAM:

#### Quantization Level Guide

| Quantization | VRAM Required | Quality | Recommended For |
| :--- | :---: | :---: | :--- |
| **Q4_K_M** | 6-8GB | ⭐⭐⭐⭐ | **Best balance** - Most users start here |
| **Q5_K_M** | 8-10GB | ⭐⭐⭐⭐½ | Better quality if you have extra VRAM |
| **Q6_K** | 10-12GB | ⭐⭐⭐⭐⭐ | Near-FP16 quality, production use |
| **Q8_0** | 12-14GB | ⭐⭐⭐⭐⭐ | Virtually identical to full precision |

**Source**: [GGUF Quantization Guide 2025](https://apatero.com/blog/gguf-quantized-models-complete-guide-2025)

#### Recommended Models for Prajnya

**1. olmOCR-2-7B (Primary Tested Model - RECOMMENDED)**

**Prajnya V5 was developed and tested with olmOCR-2-7B-1025** from AllenAI, the latest version optimized for document understanding.

Download from HuggingFace via **bartowski** (Prajnya's tested source):

```bash
# RECOMMENDED: Q5_K_M (8-10GB VRAM) - Tested configuration for Prajnya V5
wget https://huggingface.co/bartowski/allenai_olmOCR-2-7B-1025-GGUF/resolve/main/allenai_olmOCR-2-7B-1025-Q5_K_M.gguf

# Budget option: Q4_K_M (6-8GB VRAM, good balance)
wget https://huggingface.co/bartowski/allenai_olmOCR-2-7B-1025-GGUF/resolve/main/allenai_olmOCR-2-7B-1025-Q4_K_M.gguf

# High quality: Q6_K (10-12GB VRAM, better accuracy)
wget https://huggingface.co/bartowski/allenai_olmOCR-2-7B-1025-GGUF/resolve/main/allenai_olmOCR-2-7B-1025-Q6_K.gguf
```

**2. Qwen3-VL (General-purpose Vision-Language)**

```bash
# From Unsloth (Q4_K_M)
wget https://huggingface.co/unsloth/Qwen3-VL-7B-GGUF/resolve/main/Qwen3-VL-7B.Q4_K_M.gguf
```

---

### Step 3: Start llama.cpp Server

```bash
# Navigate to llama.cpp directory
cd llama.cpp

# Start server with GPU acceleration
./build/bin/llama-server \
    --model /path/to/allenai_olmOCR-2-7B-1025-Q5_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --n-gpu-layers 35 \
    --ctx-size 4096 \
    --chat-template llama3

# Windows (PowerShell)
.\build\bin\Release\llama-server.exe `
    --model C:\path\to\allenai_olmOCR-2-7B-1025-Q5_K_M.gguf `
    --host 0.0.0.0 `
    --port 8080 `
    --n-gpu-layers 35 `
    --ctx-size 4096 `
    --chat-template llama3
```

**Key Parameters:**
- `--n-gpu-layers 35`: Offload 35 layers to GPU (adjust based on VRAM)
- `--ctx-size 4096`: Context window size
- `--chat-template llama3`: Chat format (use `chatml` for some models)

---

### Step 4: Configure Prajnya for Local VLM

Update your `.env` file:

```env
# Local llama.cpp server
LOCAL_VLM_SERVER=http://localhost:8080/v1/chat/completions

# Comment out cloud APIs
# GEMINI_API_KEY=...
# OPENAI_API_KEY=...
```

Update `src/processing/strategies/config/vlm_profiles/olmocr-2-7b.yaml`:

```yaml
model_name: "allenai_olmOCR-2-7B-1025-Q5_K_M"  # Match your downloaded model
api_endpoint: "http://localhost:8080/v1/chat/completions"
model_type: "local_llama_cpp"
```

**Test Prajnya with local VLM:**
```bash
python run_Prajnya.py --pdf test_paper.pdf --preset balanced
```

---

## 3. Zotero Integration (Triple-Referee)

Enable **triple-referee metadata validation** by connecting Prajnya to your Zotero library.

### Why Zotero Integration?

- **Automatic metadata correction**: VLM says "Srnth" → Zotero API returns "Smith"
- **DOI/ISBN validation**: Cross-check against authoritative databases
- **Citation management**: Extracted papers auto-added to your Zotero library (optional)

### Step 1: Install Zotero

Download and install Zotero from https://www.zotero.org/download

### Step 2: Create Zotero Account

1. Visit https://www.zotero.org/user/register
2. Create a free account
3. Verify your email

### Step 3: Generate API Key

1. Log in to Zotero and go to **Settings**: https://www.zotero.org/settings/security
2. Scroll down to the **"Applications"** section
3. Click **"Create new private key"**
4. **Configure permissions:**
   - **Description**: "Prajnya Academic PDF Processor"
   - **Library access**: ✅ Check "Allow library access"
   - **Write access**: ✅ Check "Allow write access" (if you want Prajnya to add items)
5. Click **"Save Key"** and copy it immediately.

### Step 4: Find Your User ID (Library ID)

Your **User ID** is different from your username.
1. Visit your Zotero profile: https://www.zotero.org/[username]
2. Click on **"My Library"**
3. The URL will be: `https://www.zotero.org/users/XXXXXX/library`
4. The number `XXXXXX` is your User ID.

### Step 5: Configure Prajnya

Update your `.env` file:

```env
# Zotero Integration
ZOTERO_USER_ID=1234567          # Your numeric User ID
ZOTERO_API_KEY=a1B2c3D4e5F6g7H8 # Your API key from Step 3

# Optional: Group Library (for shared collections)
# ZOTERO_GROUP_ID=7654321
```

Enable Zotero in strategy config (`src/processing/strategies/config/research_article_strategy.yaml`):

```yaml
zotero:
  enabled: true
  cache_dir: "persistence/zotero_cache"
  primary_lookup: "doi"
```

---

## 4. Troubleshooting

### llama.cpp Server Issues

**Problem**: `Connection refused` to `localhost:8080`
- **Solution**: Ensure llama-server is running (`ps aux | grep llama-server`)
- Check firewall isn't blocking port 8080

**Problem**: `CUDA out of memory`
- **Solution**: Reduce `--n-gpu-layers` (try 25, then 15, then 10)
- Switch to smaller quantization (Q8 → Q5 → Q4)

### Zotero Integration Issues

**Problem**: `401 Unauthorized` error
- **Solution**: Verify API key is correct and check "Allow library access" permission.

**Problem**: Metadata not being validated
- **Solution**: Check `zotero.enabled: true`. Verify DOI exists in paper.

---

## 5. Performance Tuning

### llama.cpp GPU Layers

Adjust `--n-gpu-layers` based on your VRAM:

| VRAM | Recommended Layers (7B model) |
| :--- | :---: |
| 4GB | 10-15 layers |
| 6GB | 20-25 layers |
| 8GB | 30-35 layers |
| 12GB+ | 40-43 layers (full model) |

### Batch Processing Optimization

For large batches (1000+ papers):

```yaml
# research_article_strategy.yaml
concurrency:
  enabled: true
  max_concurrent: 2  # Adjust based on VRAM (1 for 8GB, 2 for 16GB, 4 for 24GB)
```
