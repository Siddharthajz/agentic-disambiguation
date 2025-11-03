# Agentic Disambiguation

A research project for evaluating Retrieval-Augmented Generation (RAG) systems on the AmbigNQ dataset—a benchmark for answering ambiguous open-domain questions.

## Overview

This project implements and compares three RAG approaches for handling ambiguous questions:

1. **Vanilla RAG**: Standard single-round retrieval and generation
2. **Iterative RAG**: Multi-round refinement with quality checking
3. **Agentic RAG**: LangGraph-based agent that detects ambiguity, generates sub-queries, and uses HyDE (Hypothetical Document Embeddings) for enhanced retrieval

All approaches use shared modular components for retrieval (sparse/dense/hybrid) and generation, with comprehensive evaluation metrics including F1, D-F1 (Disambiguation F1), nDCG, and Recall.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Building FAISS Index](#building-faiss-index)
- [Local LLM Setup (Optional)](#local-llm-setup-optional)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)

## Prerequisites

### Required Software

1. **Python 3.9+**
   - Check version: `python --version` or `python3 --version`
   - Download from [python.org](https://www.python.org/downloads/)

2. **Java 21+** (Required for PySerini/BM25 retrieval)

   **macOS:**
   ```bash
   brew install openjdk@21
   ```

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y openjdk-21-jdk
   ```

   **Verify installation:**
   ```bash
   java -version
   # Should show: openjdk version "21.x.x" or higher
   ```

3. **Git** (for cloning the repository)
   ```bash
   git --version
   ```

### System Requirements

- **Disk Space**: ~15GB free
  - Dataset and indices: ~9GB (Wikipedia BM25 indices)
  - FAISS index: ~1-2GB (if using dense retrieval)
  - Models: ~2.5GB (if using local LLM)
  - Python environment: ~1GB

- **RAM**: Minimum 8GB, recommended 16GB+
  - Dense retrieval loads large FAISS indices (~2-10GB depending on corpus size)
  - Sparse retrieval is lighter (~500MB)

- **Internet**: Required for first-time setup
  - Downloading dataset and indices
  - API calls (unless using local LLM)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/agentic-disambiguation.git
cd agentic-disambiguation
```

### Step 2: Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` at the beginning of your terminal prompt.

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If you encounter issues with PySerini installation, ensure Java 21+ is installed and `JAVA_HOME` is set:

```bash
# macOS/Linux
export JAVA_HOME=$(/usr/libexec/java_home -v 21)  # macOS
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64  # Linux

# Windows (in PowerShell)
$env:JAVA_HOME="C:\Program Files\Java\jdk-21"
```

## Environment Setup

### OpenAI API Key (for cloud-based generation)

1. Create a `.env` file in the project root:
   ```bash
   touch .env
   ```

2. Add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

3. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)

**Note:** You can skip this if you plan to use only local LLMs (see [Local LLM Setup](#local-llm-setup-optional)).

### Verify Installation

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# Test Python imports
python -c "import pyserini; import faiss; import openai; print('All imports successful!')"
```

## Dataset Preparation

### Step 1: Download AmbigNQ Dataset

The first run will automatically download the Wikipedia BM25 indices (~9GB, one-time download):

```bash
python data/ambigqa_import.py --data-dir ./data --sample-n 300
```

**What this does:**
- Downloads AmbigNQ train/dev splits from HuggingFace
- Samples 300 examples for the test set
- Downloads PySerini Wikipedia indices (first run only)
- Creates `data/ambignq_train.json`, `data/ambignq_dev.json`, `data/ambignq_test.json`

**Options:**
- `--sample-n N`: Number of test examples (default: 300, full test set: ~2000)
- `--data-dir PATH`: Directory to save data (default: ./data)

### Step 2: Verify Dataset

```bash
# Check created files
ls -lh data/

# Should show:
# ambignq_train.json
# ambignq_dev.json
# ambignq_test.json
```

## Building FAISS Index

**Required only for dense or hybrid retrieval modes.**

### Step 1: Prepare Embeddings File

Ensure you have the pre-computed embeddings file:
```bash
ls data/wiki_minilm.ndjson
```

This file should contain Wikipedia passages with pre-computed embeddings from the `all-MiniLM-L6-v2` model.

### Step 2: Build the Index

```bash
python scripts/build_faiss_index.py \
  --input-file data/wiki_minilm.ndjson \
  --output-dir ./data \
  --verify
```

**What this does:**
- Reads pre-computed embeddings from NDJSON file
- Builds FAISS index with cosine similarity (IndexFlatIP with L2-normalized vectors)
- Creates `data/ambigqa_wiki.index` (FAISS index)
- Creates `data/ambigqa_wiki_metadata.json` (document metadata)
- Runs test query if `--verify` flag is used

**Options:**
- `--input-file PATH`: Path to NDJSON embeddings file (required)
- `--output-dir PATH`: Directory to save index (default: ./data)
- `--verify`: Run test query after building

**Expected output:**
```
Loading embeddings from data/wiki_minilm.ndjson...
Loaded 1000000 documents
Building FAISS index...
Index built successfully
Saving to data/ambigqa_wiki.index...
Done! Index saved.
```

### Verify Index

```bash
python -c "
import faiss
index = faiss.read_index('data/ambigqa_wiki.index')
print(f'Index contains {index.ntotal} vectors')
print(f'Dimension: {index.d}')
"
```

## Local LLM Setup (Optional)

Run without OpenAI API using llama.cpp with Qwen models.

### Step 1: Install llama-cpp-python

```bash
pip install llama-cpp-python
```

**For M1/M2 Mac (Metal acceleration):**
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Step 2: Download Model

```bash
# List available models
python scripts/download_local_model.py --list

# Download default model (Qwen3-4B-Q4, ~2.5GB)
python scripts/download_local_model.py

# Or download higher quality model
python scripts/download_local_model.py --model qwen3-4b-q6
```

**Available models:**
- `qwen2.5-3b-q4`: Qwen2.5-3B-Instruct (Q4_K_M, ~2GB)
- `qwen2.5-3b-q6`: Qwen2.5-3B-Instruct (Q6_K, ~2GB, higher quality)
- `qwen3-4b-q4`: Qwen3-4B-Instruct-2507 (Q4_K_M, ~2.5GB) **[DEFAULT]**
- `qwen3-4b-q6`: Qwen3-4B-Instruct-2507 (Q6_K, ~3.3GB, higher quality)
- `qwen3-4b-q8`: Qwen3-4B-Instruct-2507 (Q8_0, ~4.3GB, highest quality)

Models are saved to `models/` directory.

### Step 3: Verify Local LLM

```bash
python scripts/download_local_model.py --test
```

## Quick Start

### Test the Setup

**IMPORTANT: Always activate the virtual environment first:**
```bash
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
```

Run a quick test with 5 examples:

```bash
python src/vanilla_RAG.py --retrieval-mode sparse --limit 5
```

**Expected output:**
```
Loading dataset from data/ambignq_test.json...
Loaded 5 examples
Initializing retriever...
Running Vanilla RAG on 5 examples...
100%|████████████████████| 5/5 [00:15<00:00,  3.05s/it]

Evaluation Report
=================
Examples: 5
Mean F1: 0.523
Mean D-F1: 0.400
...
```

### Single Query Test

```bash
python src/agentic_disambiguation.py \
  --single-query "When was the NBA 3-point line introduced?" \
  --retrieval-mode sparse \
  --verbose
```

This runs the full agentic pipeline on one question with detailed logging.

## Usage Examples

### Example 1: Vanilla RAG (All Retrieval Modes)

```bash
python src/vanilla_RAG.py \
  --retrieval-mode all \
  --data-path data/ambignq_test.json \
  --limit 300
```

Results automatically saved to organized structure:
- `results/vanilla/sparse/gpt-4o-mini/results.json`
- `results/vanilla/dense/gpt-4o-mini/results.json`
- `results/vanilla/hybrid/gpt-4o-mini/results.json`

### Example 2: Iterative RAG with Hybrid Retrieval

```bash
python src/iterative_RAG.py \
  --retrieval-mode hybrid \
  --max-iterations 3 \
  --limit 50
```

Results saved to: `results/iterative/hybrid/gpt-4o-mini/results_test.json`

### Example 3: Agentic RAG (Recommended)

```bash
python src/agentic_disambiguation.py \
  --retrieval-mode hybrid \
  --limit 50 \
  --verbose
```

Results saved to: `results/agentic/hybrid/gpt-4o-mini/results_test.json`

### Example 4: Using Local LLM (No API Key Required)

```bash
python src/agentic_disambiguation.py \
  --use-local-llm \
  --retrieval-mode sparse \
  --limit 10 \
  --verbose
```

Results saved to: `results/agentic/sparse/qwen3-4b-q4/results_test.json`

### Example 5: Custom Configuration

```bash
python src/vanilla_RAG.py \
  --retrieval-mode dense \
  --dense-index data/ambigqa_wiki.index \
  --dense-encoder all-MiniLM-L6-v2 \
  --dense-metadata data/ambigqa_wiki_metadata.json \
  --top-k 10 \
  --model gpt-4o-mini \
  --max-tokens 300 \
  --limit 100
```

### Compare Multiple Approaches

```bash
python src/compare_results.py \
  --vanilla results/vanilla/sparse/gpt-4o-mini/results.json \
  --iterative results/iterative/sparse/gpt-4o-mini/results.json \
  --agentic results/agentic/hybrid/gpt-4o-mini/results.json \
  --output results/comparison.json \
  --baseline vanilla
```

## Project Structure

```
agentic-disambiguation/
├── src/
│   ├── core/                      # Shared modular components
│   │   ├── data_models.py        # RetrievalResult, RAGResult dataclasses
│   │   ├── config.py             # RAGConfig for unified configuration
│   │   ├── cache.py              # RetrievalCache for caching
│   │   ├── retrievers.py         # BaseRetriever + implementations
│   │   ├── generators.py         # BaseGenerator + implementations
│   │   └── output_utils.py       # Organized output path utilities
│   │
│   ├── vanilla_RAG.py            # Baseline: Standard RAG
│   ├── iterative_RAG.py          # Baseline: Multi-round refinement
│   ├── agentic_disambiguation.py # Novel: Agentic approach with LangGraph
│   ├── evaluation.py             # Evaluation metrics (F1, D-F1, nDCG, Recall)
│   └── compare_results.py        # Compare multiple approaches
│
├── data/
│   ├── ambigqa_import.py         # Download/prepare dataset
│   ├── ambignq_train.json        # Training split
│   ├── ambignq_dev.json          # Dev split
│   ├── ambignq_test.json         # Test split (300 examples)
│   └── wiki_minilm.ndjson        # Pre-computed Wikipedia embeddings
│
├── scripts/
│   ├── build_faiss_index.py      # Build FAISS index from embeddings
│   ├── download_local_model.py   # Download local LLM models
│   ├── reorganize_results.py     # Reorganize legacy results
│   └── visualize_langgraph.py    # Visualize LangGraph workflow
│
├── results/                       # Organized experiment results
│   ├── vanilla/                   # by approach
│   │   ├── sparse/               # by retrieval mode
│   │   │   └── gpt-4o-mini/      # by model
│   │   ├── dense/
│   │   └── hybrid/
│   ├── iterative/
│   └── agentic/
├── models/                        # Local LLM models (if using)
├── .cache/                        # Retrieval cache directory
├── requirements.txt               # Python dependencies
├── .env                          # API keys (create this)
└── README.md                     # This file
```
