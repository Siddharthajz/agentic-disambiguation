# LLM Judge Quick Reference

## What It Does

Uses OpenAI LLMs to **judge if generated answers are similar to ground truth** answers by:
- Checking semantic similarity (not just token overlap)
- Assessing coverage of multiple interpretations (for ambiguous questions)
- Evaluating long-form answer quality

## Installation

Already implemented - no additional install needed. Just requires `openai` package (already in requirements.txt).

## Quick Start

### 1. Run Examples
```bash
python examples/judge_example.py
```

### 2. Judge Existing Results
```bash
python scripts/judge_results.py \
  --results-path results/agentic/hybrid/gpt-4o-mini/results.json \
  --limit 10 --verbose
```

### 3. Use in Code
```python
from src.core import OpenAIJudge, CachedJudge

judge = CachedJudge(OpenAIJudge(model="gpt-4o-mini"))

result = await judge.judge_answer_similarity(
    question="What is X?",
    generated_answer="X is Y",
    ground_truth_answers=["Y", "Y is the answer"]
)

print(f"Score: {result['similarity_score']:.2f}")
```

## Three Judgment Methods

### 1. Answer Similarity
```python
result = await judge.judge_answer_similarity(
    question: str,
    generated_answer: str,
    ground_truth_answers: List[str]
)
# Returns: similarity_score, is_similar, reasoning, coverage_score
```

### 2. Disambiguation (AmbigNQ)
```python
result = await judge.judge_disambiguation(
    question: str,
    generated_answer: str,
    ground_truth_interpretations: List[List[str]],  # List of answer sets
    dataset: "ambignq"
)
# Returns: disambiguation_score, covered/missing interpretations
```

### 3. Disambiguation (ASQA)
```python
result = await judge.judge_disambiguation(
    question: str,
    generated_answer: str,
    ground_truth_interpretations: List[Dict],  # List of qa_pairs
    dataset: "asqa"
)
# Returns: disambiguation_score, per-interpretation analysis
```

### 4. Long-Form Quality (ASQA)
```python
result = await judge.judge_long_form_answer(
    question: str,
    generated_answer: str,
    reference_answer: str
)
# Returns: quality, factuality, completeness, coherence scores
```

## Key Features

✅ **Semantic Judgment**: Understands meaning, not just tokens  
✅ **JSON Output**: Structured, easy-to-parse responses  
✅ **Caching**: Reduce API costs by 30-50x for similar answers  
✅ **Async**: Non-blocking, works with async pipelines  
✅ **Multi-dataset**: Handles AmbigNQ and ASQA formats  
✅ **Detailed Reasoning**: LLM explains each judgment  

## Files

| File | Purpose |
|------|---------|
| `src/core/judge.py` | Core module (OpenAIJudge, CachedJudge) |
| `scripts/judge_results.py` | CLI tool to judge result files |
| `examples/judge_example.py` | 5 working examples |
| `examples/judge_integration.py` | LangGraph integration pattern |
| `docs/JUDGE_GUIDE.md` | Complete documentation |
| `JUDGE_IMPLEMENTATION.md` | Implementation summary |

## Output Example

```json
{
  "similarity_score": 0.85,
  "is_similar": true,
  "matched_answer": "Paris",
  "reasoning": "Generated answer correctly identifies Paris as the capital",
  "coverage_score": 0.9,
  "coverage_reasoning": "Covers the main fact with complete information",
  "generation_time": 0.45,
  "total_tokens": 156
}
```

## Cost

- **gpt-4o-mini**: ~$0.0005 per judgment
- **100 judgments**: ~$0.05
- **1000 judgments**: ~$0.50 (base) or ~$0.10 (with caching)

## Integration Options

### Option 1: Real-time (LangGraph Node)
```python
workflow.add_node("judge_answer", self.judge_answer_node)
workflow.add_edge("synthesize_answer", "judge_answer")
workflow.add_edge("judge_answer", END)
```

### Option 2: Batch Post-Processing
```python
results = await framework.run_batch(test_data)
judgments = await evaluator.judge_results(results)
```

### Option 3: With Regeneration
```python
if judgment['similarity_score'] < 0.6:
    # Trigger regeneration
    state['requires_regeneration'] = True
```

## Configuration

Add to CLI:
```bash
--use-judge                        # Enable judge
--judge-model gpt-4o-mini         # Judge model
--judge-quality-threshold 0.6     # Regen if below this
--judge-max-regenerations 2       # Max regen attempts
```

## Common Use Cases

### Case 1: Evaluate Existing Results
```bash
python scripts/judge_results.py --results-path results.json --save-judgments judgments.json
```

### Case 2: Compare with Automatic Metrics
```python
# In evaluation.py, alongside F1/ROUGE-L
automatic_f1 = compute_f1(generated, ground_truth)
judge_score = await judge.judge_answer_similarity(...)
correlation = compute_correlation(automatic_f1, judge_score)
```

### Case 3: Quality Filtering
```python
results = await framework.run_batch(test_data)
high_quality = [r for r in results if r['metadata']['judge_similarity']['similarity_score'] > 0.7]
```

### Case 4: Error Analysis
```python
# Find low-scoring answers
low_scores = [r for r in results if r['metadata']['judge_similarity']['similarity_score'] < 0.5]
# Analyze why with judge reasoning
for r in low_scores:
    print(r['metadata']['judge_similarity']['reasoning'])
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "OPENAI_API_KEY not set" | `export OPENAI_API_KEY=your-key` |
| "API rate limit" | Use caching, reduce concurrency |
| "Poor judge scores" | Check ground truth format, adjust prompts |
| "Slow evaluation" | Enable caching, batch process |

## Next Steps

1. **Try examples**: `python examples/judge_example.py`
2. **Judge your results**: `python scripts/judge_results.py --results-path results.json`
3. **Integrate**: Add judge node to pipeline
4. **Analyze**: Compare judge vs automatic metrics
5. **Iterate**: Tune judge prompts for your use case

## Documentation

- Full guide: `docs/JUDGE_GUIDE.md`
- Implementation details: `JUDGE_IMPLEMENTATION.md`
- Integration patterns: `examples/judge_integration.py`
- Working examples: `examples/judge_example.py`

---

**Questions?** Check the docs or run the examples!
