# Judge Batch Results Summary

## Batch Execution Completed ✓

**Date:** December 3, 2025  
**Judge Model:** GPT-4o-mini  
**Total Datasets:** 20 (10 AmbigNQ + 10 ASQA)  
**Total Results Judged:** 6,000 (300 per dataset)

### Files Generated

- **20 Judgment Files** (`*_judgments.json`): Full LLM judgment results with similarity, disambiguation, and long-form scores
- **Comparison Table** (`judge_comparison_table.csv`): Metrics summary across all datasets
- **JSON Summary** (`judge_comparison.json`): Structured metrics for programmatic access

### Key Findings

#### By RAG Approach (AmbigNQ)

| Approach   | Avg Similarity | Avg Disambiguation | Coverage Rate |
|------------|----------------|--------------------|---------------|
| **Agentic** | 0.469          | 0.493              | 0.48          |
| **Iterative** | 0.468        | 0.485              | 0.47          |
| **Vanilla** | 0.438         | 0.442              | 0.42          |

**Winner:** Agentic approach performs best across all metrics

#### Top Performers (AmbigNQ)

1. **ambignq_agentic_hybrid_gpt-4o-mini**: Sim=0.63, Dis=0.57, Cov=0.55
2. **ambignq_iterative_sparse_gpt-4o-mini**: Sim=0.57, Dis=0.48, Cov=0.47
3. **ambignq_vanilla_hybrid_gpt-4o-mini**: Sim=0.54, Dis=0.49, Cov=0.47

#### ASQA Similarity Scores

- Best: **asqa_agentic_hybrid_gpt-4o-mini** (0.50)
- Avg: 0.41 (all ASQA)
- Note: ASQA does not have disambiguation judgments (no multiple interpretations in this format)

### Metrics Explained

- **Similarity Mean**: Semantic similarity to ground truth (0-1). Higher = better alignment.
- **Dis Mean**: How well answers address multiple interpretations (0-1). Only for AmbigNQ.
- **Cov Rate**: Coverage of interpretations across dataset (0-1). Only for AmbigNQ.
- **Sim Std**: Consistency of answers. Lower = more consistent.

### Data Structure

Each judgment file contains:

```json
{
  "judge_model": "gpt-4o-mini",
  "results_path": "ambignq/agentic/hybrid/gpt-4o-mini/results.json",
  "judgments": [
    {
      "question": "...",
      "generated_answer": "...",
      "ground_truth_answers": ["..."],
      "qa_pairs": [...],
      "dataset": "ambignq",
      "similarity": { "similarity_score": 0.X, ... },
      "disambiguation": { "total_interpretations": N, ... }
    }
  ],
  "original_queries": [...],
  "aggregate_metrics": {
    "total_judged": 300,
    "similarity_scores": { "mean": 0.X, "std": 0.Y, ... },
    "disambiguation_scores": { "mean": 0.X, "coverage_rate": 0.Y, ... }
  }
}
```

### Files Available

**Judgment Files** (19 total, 300 results each):
- `ambignq_*_judgments.json` (10 files)
- `asqa_*_judgments.json` (9 files)

**Comparison Files**:
- `judge_comparison_table.csv` - Quick reference table
- `judge_comparison.json` - Full metrics by dataset and approach

**Documentation**:
- `JUDGE_METRICS_GUIDE.md` - Detailed metric explanations
- `JUDGE_BATCH_RESULTS.md` - This file

### Next Steps

- Analyze by retrieval method (sparse vs dense vs hybrid)
- Compare with existing evaluation metrics (F1, ROUGE, etc.)
- Investigate failure cases where similarity_score is low
- Fine-tune prompts for higher disambiguation coverage on ASQA
