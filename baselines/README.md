# RAG Baselines for AmbigNQ

This directory contains a Retrieval-Augmented Generation (RAG) baseline for the AmbigNQ dataset with sparse, dense, and hybrid retrieval plus a comprehensive evaluator.

## What's here

- `vanilla_RAG.py`: End-to-end RAG pipeline (BM25, FAISS dense, and hybrid) with async OpenAI generation and built-in evaluation.
- `evaluation.py`: Metrics for answer quality (F1), ambiguity handling (D-F1), retrieval quality (nDCG@k, Recall@k), and efficiency.

## Prerequisites

- Python 3.9+
- Java 21 (required by PySerini)
- An OpenAI API key

## Setup

From the project root:

```bash
pip install -r requirements.txt
```

Install Java 21 if needed:
- macOS: `brew install openjdk@21`
- Ubuntu/Debian: `sudo apt-get install -y openjdk-21-jdk`

Create a `.env` in the project root and add your API key:

```
OPENAI_API_KEY=sk-...
```

Note: On first run, PySerini will automatically download prebuilt Wikipedia indexes (~9GB). Ensure you have enough disk space and a stable internet connection.

## Quick start

Run from the project root so default paths resolve correctly:

```bash
python baselines/vanilla_RAG.py --retrieval-mode sparse --limit 5
```

This will process 5 examples using BM25, print an evaluation report, and write results under `results/`.

## Usage

`vanilla_RAG.py` supports four retrieval modes: `sparse`, `dense`, `hybrid`, or `all`.

### Sparse retrieval (BM25)
```bash
python baselines/vanilla_RAG.py \
  --retrieval-mode sparse \
  --data-path data/ambignq_test.json \
  --output-path results/sparse_rag_results.json \
  --sparse-index wikipedia-dpr \
  --top-k 5 \
  --model gpt-4o-mini
```

### Dense retrieval (FAISS + BPR encoder)
```bash
python baselines/vanilla_RAG.py \
  --retrieval-mode dense \
  --data-path data/ambignq_test.json \
  --output-path results/dense_rag_results.json \
  --dense-index wikipedia-dpr-100w.bpr-single-nq \
  --dense-encoder castorini/bpr-nq-question-encoder \
  --top-k 5 \
  --model gpt-4o-mini
```

### Hybrid retrieval (RRF of sparse + dense)
```bash
python baselines/vanilla_RAG.py \
  --retrieval-mode hybrid \
  --data-path data/ambignq_test.json \
  --output-path results/hybrid_rag_results.json \
  --sparse-index wikipedia-dpr \
  --dense-index wikipedia-dpr-100w.bpr-single-nq \
  --dense-encoder castorini/bpr-nq-question-encoder \
  --top-k 5 \
  --model gpt-4o-mini
```

### Run all modes in one go
This runs `sparse`, `dense`, and `hybrid` sequentially and saves three files:
`rag_results_sparse.json`, `rag_results_dense.json`, and `rag_results_hybrid.json`.

```bash
python baselines/vanilla_RAG.py \
  --retrieval-mode all \
  --data-path data/ambignq_test.json \
  --output-path results/rag_results.json \
  --top-k 5 \
  --model gpt-4o-mini \
  --concurrency 10
```

## Arguments

- `--data-path` (str): Path to AmbigNQ JSON. Default: `data/ambignq_test.json`
- `--output-path` (str): Where to save results JSON. Default: `results/rag_results.json`
- `--retrieval-mode` (str): `sparse`, `dense`, `hybrid`, or `all`. Default: `all`
- `--sparse-index` (str): PySerini sparse index. Default: `wikipedia-dpr`
- `--dense-index` (str): PySerini FAISS dense index. Default: `wikipedia-dpr-100w.bpr-single-nq`
- `--dense-encoder` (str): Query encoder for dense retrieval. Default: `castorini/bpr-nq-question-encoder`
- `--top-k` (int): Number of documents to retrieve. Default: `5`
- `--model` (str): OpenAI model (e.g., `gpt-4o-mini`, `gpt-4.1-mini`). Default: `gpt-4o-mini`
- `--max-tokens` (int): Max tokens in generated answer. Default: `200`
- `--limit` (int): Process only the first N examples. Default: all
- `--concurrency` (int): Concurrent OpenAI requests. Default: `10`

## Output

Each run writes a JSON with:
- `config`: The CLI arguments used
- `aggregate_metrics`: Overall evaluation from `evaluation.py`
- `results`: Per-example info (retrieved docs, generated answer, timings, tokens, metrics)

When `--retrieval-mode all` is used, three files are produced by appending the mode to the stem of `--output-path`.

## Evaluation summary

The built-in evaluator reports:
- Answer Quality: token-level F1
- Ambiguity Handling: Disambiguation F1 (D-F1) and coverage (≥ 0.5 F1 counts as covered)
- Retrieval Quality: nDCG@k and Recall@k (k = `--top-k`)
- Efficiency: retrieval time, generation time, total tokens

## PySerini indexes

Common prebuilt indexes used here:
- `wikipedia-dpr` (sparse BM25 over DPR passages)
- `wikipedia-dpr-100w.bpr-single-nq` (dense FAISS + BPR NQ encoder)

PySerini will download these on first use.

## Troubleshooting

### Java not found
Install Java 11+ and ensure it is on your PATH.
- macOS: `brew install openjdk@11`
- Ubuntu/Debian: `sudo apt-get install -y openjdk-11-jdk`

### API key not set
Error:
```
ValueError: OPENAI_API_KEY environment variable not set
```
Create a `.env` at the project root with `OPENAI_API_KEY=...` or export it in your shell.

### Index download or loading issues
Ensure:
- 10GB+ free disk space
- Stable internet connection
- Java 11+ installed

### Rate limiting
Reduce `--concurrency`, set `--limit` while debugging, or run again after a pause.

## Citation

If you use this baseline, please cite AmbigQA:

```bibtex
@inproceedings{min2020ambigqa,
  title={AmbigQA: Answering Ambiguous Open-domain Questions},
  author={Min, Sewon and Michael, Julian and Hajishirzi, Hannaneh and Zettlemoyer, Luke},
  booktitle={EMNLP},
  year={2020}
}
```
