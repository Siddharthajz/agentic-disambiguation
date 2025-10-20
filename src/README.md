# RAG Systems for Ambiguous Question Answering

This directory contains implementations of different RAG approaches for handling ambiguous questions on the AmbigNQ dataset.

## Architecture Overview

The codebase uses a **modular architecture** with shared components across all approaches:

```
src/
├── core/                          # Shared modular library
│   ├── __init__.py               # Exports all core components
│   ├── data_models.py            # RetrievalResult, RAGResult dataclasses
│   ├── config.py                 # RAGConfig for unified configuration
│   ├── cache.py                  # RetrievalCache for caching retrieval results
│   ├── retrievers.py             # Base + implementations (Sparse, Dense, Hybrid)
│   └── generators.py             # Base + implementations (OpenAI, HyDE)
│
├── vanilla_RAG.py                # Baseline 1: Standard RAG
├── iterative_RAG.py              # Baseline 2: Multi-round refinement
├── agentic_disambiguation.py     # Novel: Agentic approach with sub-queries + HyDE
│
├── evaluation.py                 # Comprehensive evaluation metrics
├── compare_results.py            # Utility for comparing approaches
└── README.md                     # This file
```

## Key Features

### 1. Modular Core Library (`core/`)

**Retrievers** (`core/retrievers.py`):
- `BaseRetriever`: Abstract base class
- `SparseRetriever`: BM25 via PySerini/Lucene
- `DenseRetriever`: FAISS with neural encoders
- `HybridRetriever`: RRF (Reciprocal Rank Fusion)
- `create_retriever()`: Factory function for easy instantiation

**Generators** (`core/generators.py`):
- `BaseGenerator`: Abstract base class
- `OpenAIGenerator`: Async OpenAI API with batching
- `HyDEGenerator`: Hypothetical Document Embeddings

**Caching** (`core/cache.py`):
- `RetrievalCache`: Automatic caching of retrieval results
- Deterministic: Same query → Same cached results
- Configurable cache directory

**Configuration** (`core/config.py`):
- `RAGConfig`: Centralized configuration management
- Easy creation from argparse arguments

### 2. Optimization Features

**Performance**:
- ✅ Retrieval caching (avoid redundant searches)
- ✅ Async generation with concurrency control
- ✅ Lazy loading of indices (load only what's needed)
- ✅ Memory cleanup between modes
- ✅ Batch processing with progress bars

**Extensibility**:
- ✅ Easy to add new retriever types
- ✅ Easy to add new generator types
- ✅ Shared evaluation metrics
- ✅ Pluggable components via dependency injection

## Approaches

### 1. Vanilla RAG (Baseline 1)

**File**: `vanilla_RAG.py`

**Pipeline**:
```
Question → Retrieve (BM25/Dense/Hybrid) → Generate Answer → Evaluate
```

**Features**:
- Single-round retrieval
- Direct answer generation
- Supports all three retrieval modes

**Usage**:
```bash
# Sparse retrieval
python src/vanilla_RAG.py \
  --retrieval-mode sparse \
  --data-path data/ambignq_test.json \
  --output-path results/vanilla_sparse.json \
  --limit 50

# Run all modes
python src/vanilla_RAG.py --retrieval-mode all --limit 50
```

### 2. Iterative RAG (Baseline 2)

**File**: `iterative_RAG.py`

**Pipeline**:
```
Question → [Retrieve → Generate → Check] × N iterations → Best Answer
```

**Features**:
- Multi-round refinement
- Query reformulation based on previous answers
- Configurable max iterations
- Accumulates documents across iterations

**Usage**:
```bash
python src/iterative_RAG.py \
  --retrieval-mode hybrid \
  --max-iterations 3 \
  --limit 50 \
  --output-path results/iterative_hybrid.json
```

**TODOs**:
- [ ] Implement LLM-based query reformulation
- [ ] Add confidence scoring for early stopping
- [ ] Implement answer quality checks

### 3. Agentic Disambiguation (Novel Approach)

**File**: `agentic_disambiguation.py`

**Pipeline**:
```
Question → Detect Ambiguity → Generate Sub-queries
  → HyDE Documents → Enhanced Retrieval → Multi-answer Generation
  → Synthesize Comprehensive Answer
```

**Features**:
- Sub-query decomposition for multiple interpretations
- HyDE (Hypothetical Document Embeddings) for improved retrieval
- Combines retrieval from sub-queries and HyDE docs
- Designed for ambiguity-aware answer generation

**Usage**:
```bash
python src/agentic_disambiguation.py \
  --retrieval-mode hybrid \
  --limit 50 \
  --output-path results/agentic_hybrid.json
```

**TODOs**:
- [ ] Implement LLM-based sub-query generation
- [ ] Integrate LangGraph for agent orchestration
- [ ] Add ambiguity detection module
- [ ] Implement answer synthesis strategy

## Evaluation

All approaches use the same comprehensive evaluation metrics:

**Answer Quality**:
- F1 Score (token-level overlap with ground truth)

**Ambiguity Handling**:
- D-F1 (Disambiguation F1): Coverage of multiple interpretations
- Coverage Rate: Fraction of interpretations covered

**Retrieval Quality**:
- nDCG@k: Ranking quality
- Recall@k: Fraction of relevant docs retrieved

**Efficiency**:
- Retrieval time, generation time, total time
- Token usage (for cost tracking)
- Latency percentiles (p50, p95, p99)

## Comparing Results

Use `compare_results.py` to compare all approaches:

```bash
python src/compare_results.py \
  --vanilla results/vanilla_rag_results_sparse.json \
  --iterative results/iterative_rag_results_sparse.json \
  --agentic results/agentic_disambiguation_results_hybrid.json \
  --output results/comparison_summary.json \
  --baseline vanilla
```

This generates:
- Comparative table across all metrics
- Improvement percentages over baseline
- JSON summary with best approach per dimension

## Development Workflow

### Adding a New Retriever

1. Create a new class in `core/retrievers.py`:
```python
class MyRetriever(BaseRetriever):
    def retrieve(self, query: str, k: Optional[int] = None):
        # Your implementation
        pass

    def get_cache_params(self):
        return {"my_param": self.my_param}
```

2. Update `create_retriever()` factory function
3. Use in any approach: `retriever = create_retriever(mode="my_mode")`

### Adding a New Generator

1. Create a new class in `core/generators.py`:
```python
class MyGenerator(BaseGenerator):
    async def generate(self, question, contexts):
        # Your implementation
        pass
```

2. Use in any approach: `generator = MyGenerator(...)`

### Adding a New Approach

1. Create new file: `src/my_approach.py`
2. Import from `core`: `from core import RAGConfig, create_retriever, OpenAIGenerator`
3. Implement your pipeline using shared components
4. Add evaluation and CLI arguments

## Testing

Run a quick test with 5 examples:

```bash
# Test vanilla RAG
python src/vanilla_RAG.py --retrieval-mode sparse --limit 5

# Test iterative RAG
python src/iterative_RAG.py --retrieval-mode sparse --limit 5 --max-iterations 2

# Test agentic (when implemented)
python src/agentic_disambiguation.py --retrieval-mode hybrid --limit 5
```

## Configuration

All approaches accept these common arguments:

**Data**:
- `--data-path`: Path to test data (default: `data/ambignq_test.json`)
- `--output-path`: Output results path
- `--limit`: Limit number of examples (for testing)

**Retrieval**:
- `--retrieval-mode`: `sparse`, `dense`, `hybrid`, or `all`
- `--sparse-index`: BM25 index name
- `--dense-index`: FAISS index name
- `--dense-encoder`: Query encoder model
- `--top-k`: Number of documents to retrieve (default: 5)

**Generation**:
- `--model`: OpenAI model (default: `gpt-4o-mini`)
- `--max-tokens`: Max tokens per generation (default: 200)

**Performance**:
- `--concurrency`: Concurrent OpenAI requests (default: 10)
- `--use-cache` / `--no-cache`: Enable/disable caching

## Caching

The retrieval cache is stored in `.cache/retrieval/` by default. To clear cache:

```bash
rm -rf .cache/retrieval/
```

## Future Enhancements

1. **Iterative RAG**:
   - LLM-based query reformulation
   - Confidence-based early stopping
   - Dynamic iteration count

2. **Agentic Disambiguation**:
   - LangGraph integration for agent orchestration
   - Sub-query generation with LLM
   - Multi-interpretation answer synthesis
   - Ambiguity detection and scoring

3. **General**:
   - Add more retriever types (ColBERT, BM42, rerankers)
   - Implement streaming generation
   - Add more generator backends (Anthropic, local models)
   - Distributed evaluation for large datasets

## References

- **AmbigNQ Dataset**: Min et al., 2020
- **HyDE**: Gao et al., 2022
- **RRF**: Cormack et al., 2009
- **PySerini**: Lin et al., 2021
