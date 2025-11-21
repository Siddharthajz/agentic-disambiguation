# Agentic Disambiguation for Ambiguous Question Answering

A research project investigating agentic RAG (Retrieval-Augmented Generation) approaches for handling ambiguous open-domain questions. This work evaluates how LLM-powered agents can detect and resolve question ambiguity through sub-query decomposition, hypothetical document generation, and structured multi-interpretation synthesis.

## Research Motivation

Open-domain question answering systems frequently encounter **ambiguous questions**—queries with multiple valid interpretations that require different answers. For example:

> *"When did the US break away from England?"*

This question has multiple valid interpretations:
- **Declaration of Independence**: July 4, 1776
- **Treaty of Paris (formal recognition)**: September 3, 1783
- **End of Revolutionary War**: 1781 (Yorktown)

Standard RAG systems typically return a single answer, missing the inherent ambiguity. This project explores **agentic approaches** that explicitly detect ambiguity and provide comprehensive coverage of all plausible interpretations.

## Key Contributions

1. **Coherence-Based Ambiguity Detection**: Using document embedding clustering and silhouette scores to distinguish between:
   - **Aleatoric uncertainty** (genuine ambiguity with distinct interpretations)
   - **Epistemic uncertainty** (lack of knowledge/evidence)

2. **Cluster-Guided Sub-Query Generation**: Generating interpretation-specific sub-queries from document clusters rather than purely LLM-based decomposition

3. **HyDE-Enhanced Retrieval**: Hypothetical Document Embeddings for improved semantic retrieval per interpretation

4. **Structured Multi-Intent Synthesis**: JSON-structured outputs with explicit intent labeling and confidence scoring

## Dataset: AmbigNQ

This project uses [AmbigNQ](https://nlp.cs.washington.edu/ambigqa/) (Ambiguous Natural Questions), a benchmark derived from Google's Natural Questions dataset where annotators identified questions with multiple valid interpretations.

### Dataset Structure

```json
{
  "question": "When was the nba 3 point line introduced?",
  "annotations": [
    {
      "type": "multipleQAs",
      "qaPairs": [
        {"question": "When was the NBA 3-point line introduced?", "answer": ["1979"]},
        {"question": "When was the ABA 3-point line introduced?", "answer": ["1967"]}
      ]
    }
  ],
  "nq_answer": ["1979"],
  "viewed_doc_titles": ["Three-point field goal"]
}
```

### Obtaining the Data

```bash
python data/ambigqa_import.py --data-dir ./data --sample-n 300
```

This downloads the AmbigNQ dataset from the University of Washington NLP group and creates:
- `ambignq_train.json`: Training split
- `ambignq_dev.json`: Development split
- `ambignq_test.json`: Sampled test set (300 examples by default)

## Agentic Pipeline Architecture

The core contribution is a **LangGraph-based agentic pipeline** that orchestrates multiple specialized nodes for ambiguity handling:

```
                    ┌─────────────────────┐
                    │   Input Question    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Ambiguity Detection │
                    │  (Coherence Check)  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼────────┐       │       ┌────────▼────────┐
     │   Unambiguous   │       │       │    Ambiguous/   │
     │                 │       │       │    Uncertain    │
     └────────┬────────┘       │       └────────┬────────┘
              │                │                │
              │         ┌──────▼──────┐         │
              │         │   Simple    │         │
              │         │  Retrieval  │         │
              │         └──────┬──────┘         │
              │                │     ┌──────────▼──────────┐
              │                │     │ Sub-Query Generation│
              │                │     │  (Cluster-Guided)   │
              │                │     └──────────┬──────────┘
              │                │                │
              │                │     ┌──────────▼──────────┐
              │                │     │   HyDE Generation   │
              │                │     │  (Per Sub-Query)    │
              │                │     └──────────┬──────────┘
              │                │                │
              │                │     ┌──────────▼──────────┐
              │                │     │ Enhanced Retrieval  │
              │                │     │ (Sub-Queries+HyDE)  │
              │                │     └──────────┬──────────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Answer Synthesis   │
                    │ (Structured Output) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Multi-Intent      │
                    │   JSON Response     │
                    └─────────────────────┘
```

### Node 1: Ambiguity Detection with Coherence Check

Rather than relying solely on LLM prompting to detect ambiguity, this node uses **embedding-based coherence analysis**:

```python
# Cluster retrieved documents
kmeans = KMeans(n_clusters=2).fit(doc_embeddings)

# Calculate metrics
centroid = np.mean(embeddings, axis=0)
variance = euclidean_distances(embeddings, [centroid]).mean()
separability = silhouette_score(embeddings, kmeans.labels_)

# Classification logic
if variance > THRESHOLD and separability > THRESHOLD:
    status = "Ambiguous"  # Distinct clusters = multiple interpretations
elif variance > THRESHOLD:
    status = "Uncertain"  # High variance, no structure = epistemic failure
else:
    status = "Unambiguous"
```

**Key insight**: High document variance with high cluster separability indicates genuine ambiguity (aleatoric), while high variance with low separability indicates insufficient/conflicting evidence (epistemic).

### Node 2: Cluster-Guided Sub-Query Generation

For ambiguous questions, sub-queries are generated from document clusters rather than pure LLM imagination:

1. **Cluster documents** using K-Means on embeddings
2. **Relevance pruning**: Filter clusters with low cosine similarity to the original query
3. **Generate sub-queries**: For each valid cluster, prompt the LLM to formulate a specific question representing that interpretation

This grounds sub-query generation in actual retrieved evidence.

### Node 3: HyDE (Hypothetical Document Embeddings)

For each sub-query, generate a hypothetical Wikipedia passage that would answer it:

```
Question: When was the NBA 3-point line introduced?

Hypothetical Document:
"The NBA adopted the three-point line for the 1979-80 season,
borrowing the concept from the ABA which had used it since 1967..."
```

These hypothetical documents are then used as additional retrieval queries, improving semantic matching for specific interpretations.

### Node 4: Enhanced Retrieval

Retrieval is performed for both:
- Original sub-queries
- Generated HyDE documents

Results are **deduplicated** by document ID and **re-ranked** by score, ensuring diverse coverage without redundancy.

### Node 5: Structured Answer Synthesis

The final answer uses structured JSON output:

```json
{
  "intents": [
    {
      "intent_label": "NBA Introduction",
      "confidence": 0.9,
      "key_facts": ["1979-80 season", "borrowed from ABA"]
    },
    {
      "intent_label": "ABA Introduction",
      "confidence": 0.8,
      "key_facts": ["1967", "original three-point line"]
    }
  ],
  "synthesis": "The three-point line was introduced in the NBA for the 1979-80 season, though it originated in the ABA in 1967...",
  "concise_answer": "1979 (NBA) / 1967 (ABA)"
}
```

## Retrieval Components

### Sparse Retrieval (BM25)
- **Implementation**: PySerini wrapping Apache Lucene
- **Index**: `wikipedia-dpr` (Wikipedia passages from DPR project)
- **Characteristics**: Fast, keyword-based, ~50ms per query

### Dense Retrieval (FAISS)
- **Implementation**: FAISS with sentence-transformers
- **Encoder**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Index**: Custom-built from Simple Wikipedia embeddings
- **Characteristics**: Semantic matching, ~100ms per query after warmup

> **Note on corpus selection**: The full Wikipedia DPR corpus (~21M passages) requires approximately **80GB of active RAM** to load the FAISS index, making it infeasible for resource-constrained environments. This project uses **Simple Wikipedia** as a substitute for dense retrieval, which provides adequate coverage for evaluation while remaining tractable (~2-4GB RAM). The sparse retrieval (BM25) still uses the full Wikipedia index via PySerini's memory-mapped streaming approach.

### Hybrid Retrieval (RRF)
- **Algorithm**: Reciprocal Rank Fusion
- **Formula**: `score = 1/(60 + rank_sparse) + 1/(60 + rank_dense)`
- **Rationale**: Combines keyword precision with semantic recall

## Evaluation Metrics

### Disambiguation F1 (D-F1) — Primary Metric

D-F1 measures coverage of all plausible interpretations:

```
D-F1 = interpretations_covered / total_interpretations
```

An interpretation is "covered" if the generated answer achieves F1 ≥ 0.5 with any valid answer for that interpretation.

**Example**:
- Question has 3 annotated interpretations
- System answer covers 2 of them
- D-F1 = 2/3 = 0.667

### Answer Quality (F1)

Token-level F1 score after normalization (lowercase, remove articles/punctuation). Maximum F1 across all valid reference answers.

### Retrieval Quality

- **nDCG@k**: Normalized Discounted Cumulative Gain
- **Recall@k**: Fraction of relevant documents retrieved

## Experimental Results

Comparison on 300 AmbigNQ test examples (hybrid retrieval, GPT-4o-mini):

| Approach | Mean F1 | Mean D-F1 | Coverage Rate | Tokens |
|----------|---------|-----------|---------------|--------|
| Vanilla RAG | 0.545 | 0.375 | 28.1% | 247K |
| **Agentic** | **0.592** | **0.394** | **29.3%** | 290K |

**Key findings**:
- Agentic approach improves F1 by ~8.6% and D-F1 by ~5.1%
- Trade-off: Higher token usage due to sub-query generation and HyDE
- Epistemic uncertainty detection helps identify questions where retrieval fails

## Installation

### Prerequisites

- Python 3.9+
- Java 21+ (required for PySerini/BM25)
- ~15GB disk space

### Setup

```bash
# Clone repository
git clone https://github.com/aravadikesh/agentic-disambiguation.git
cd agentic-disambiguation

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Download dataset (first run downloads ~9GB Wikipedia indices)
python data/ambigqa_import.py --data-dir ./data --sample-n 300
```

### Building FAISS Index (for dense/hybrid retrieval)

```bash
python scripts/build_faiss_index.py \
  --input-file data/wiki_minilm.ndjson \
  --output-dir ./data \
  --verify
```

## Usage

### Running the Agentic Pipeline

```bash
# Quick test (5 examples)
python src/agentic_disambiguation.py \
  --retrieval-mode sparse \
  --limit 5 \
  --verbose

# Full experiment
python src/agentic_disambiguation.py \
  --retrieval-mode hybrid \
  --limit 300

# Single query (debugging)
python src/agentic_disambiguation.py \
  --single-query "When did the US break away from England?" \
  --retrieval-mode hybrid \
  --verbose
```

### Local LLM (No API Required)

```bash
# Download model (~2.5GB)
python scripts/download_local_model.py --model qwen3-4b-q4

# Run with local inference
python src/agentic_disambiguation.py \
  --use-local-llm \
  --retrieval-mode sparse \
  --limit 10
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--retrieval-mode` | `sparse`, `dense`, `hybrid`, or `all` | `hybrid` |
| `--limit` | Number of examples to process | None (all) |
| `--verbose` | Detailed logging per step | False |
| `--single-query` | Run on single question | None |
| `--use-local-llm` | Use llama.cpp instead of API | False |
| `--top-k` | Documents to retrieve | 5 |
| `--model` | OpenAI model | `gpt-4o-mini` |

## Project Structure

```
agentic-disambiguation/
├── src/
│   ├── agentic_disambiguation.py   # Main agentic pipeline (LangGraph)
│   ├── evaluation.py               # F1, D-F1, nDCG, Recall metrics
│   ├── compare_results.py          # Cross-approach comparison
│   ├── vanilla_RAG.py              # Baseline: standard RAG
│   ├── iterative_RAG.py            # Baseline: multi-round RAG
│   └── core/
│       ├── retrievers.py           # Sparse/Dense/Hybrid retrievers
│       ├── generators.py           # OpenAI, HyDE, LlamaCpp generators
│       ├── data_models.py          # RetrievalResult, RAGResult
│       ├── config.py               # RAGConfig dataclass
│       └── cache.py                # Retrieval caching
├── data/
│   ├── ambigqa_import.py           # Dataset download script
│   └── ambignq_*.json              # Dataset files
├── scripts/
│   ├── build_faiss_index.py        # FAISS index builder
│   └── download_local_model.py     # Local LLM downloader
├── results/                        # Experiment outputs
│   └── {approach}/{mode}/{model}/
└── requirements.txt
```

## Output Format

Results are saved as JSON with this structure:

```json
{
  "config": { "retrieval_mode": "hybrid", "model": "gpt-4o-mini", ... },
  "aggregate_metrics": {
    "num_examples": 300,
    "mean_f1": 0.592,
    "mean_d_f1": 0.394,
    "coverage_rate": 0.293,
    "epistemic_uncertainty_rate": 0.12
  },
  "results": [
    {
      "question": "...",
      "generated_answer": "...",
      "evaluation": { "f1_score": 0.67, "d_f1": 0.5, ... },
      "metadata": {
        "ambiguity_status": "Ambiguous",
        "subqueries": ["...", "..."],
        "hyde_documents": { "subquery1": "..." },
        "intents": [{ "intent_label": "...", "confidence": 0.9 }]
      }
    }
  ]
}
```

## References

- **AmbigNQ Dataset**: Min et al., [AmbigQA: Answering Ambiguous Open-domain Questions](https://arxiv.org/abs/2004.10645), EMNLP 2020
- **HyDE**: Gao et al., [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496), ACL 2023
- **Reciprocal Rank Fusion**: Cormack et al., [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://dl.acm.org/doi/10.1145/1571941.1572114), SIGIR 2009
- **LangGraph**: [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
