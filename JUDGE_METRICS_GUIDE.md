# Judge Metrics Guide

This document explains the semantic judgment metrics computed by the LLM judge when evaluating RAG results.

## Overview

The judge evaluates RAG generated answers using three complementary judgment methods, each designed to assess different aspects of answer quality:

1. **Similarity Judgment** — Semantic similarity to ground truth answers
2. **Disambiguation Judgment** — Coverage of multiple question interpretations
3. **Long-Form Judgment** — Quality of complete answer explanations

---

## Similarity Judgment

### Purpose
Evaluates whether the generated answer is semantically similar to the ground truth answers, regardless of exact wording.

### Metrics

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `similarity_score` | Float | [0.0, 1.0] | Semantic similarity between generated and ground truth. 0 = completely different, 1 = equivalent meaning. |
| `is_similar` | Boolean | - | Whether the generated answer matches any ground truth answer (typically threshold >= 0.5 similarity). |
| `matched_answer` | String or null | - | The ground truth answer that was matched, if any. `null` if no match found. |
| `coverage_score` | Float | [0.0, 1.0] | Proportion of ground truth answers that are covered or addressed by the generated answer. |
| `reasoning` | String | - | Explanation of how the similarity judgment was made. |

### Aggregate Statistics

Computed across all judgments for a dataset:

| Statistic | Description |
|-----------|-------------|
| `mean` | Average similarity score across all answers. Higher is better. |
| `median` | Median similarity score. Robust to outliers. |
| `std` | Standard deviation. Lower indicates more consistent performance. |
| `min` | Minimum similarity score (worst case). |
| `max` | Maximum similarity score (best case). |
| `count` | Number of answers with similarity judgment. |

### Interpretation

- **High mean (> 0.7)**: RAG system generates answers close to ground truth meanings.
- **Low mean (< 0.4)**: Generated answers differ significantly from expected answers, may contain errors or hallucinations.
- **High std (> 0.3)**: Inconsistent performance — some answers excellent, others poor.
- **Low std (< 0.15)**: Consistent performance — predictable behavior across questions.

---

## Disambiguation Judgment

### Purpose
Evaluates whether the generated answer addresses multiple interpretations of an ambiguous question.

Used when a question has multiple valid interpretations (common in **AmbigNQ** and some **ASQA** questions).

### Metrics

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `disambiguation_score` | Float | [0.0, 1.0] | Overall score for how well the answer handles ambiguity. 0 = ignores all interpretations, 1 = clearly addresses all. |
| `interpretations_covered` | Integer | [0, N] | Number of question interpretations that are addressed by the answer. |
| `total_interpretations` | Integer | [1, N] | Total number of valid interpretations for this question. |
| `coverage_percentage` | Float | [0.0, 1.0] | Fraction of interpretations covered: `interpretations_covered / total_interpretations`. |
| `identified_interpretations` | List[String] | - | Which interpretations the judge identified in the generated answer. |
| `missing_interpretations` | List[String] | - | Interpretations that were not addressed. |
| `reasoning` | String | - | Explanation of which interpretations were covered and which missed. |

### Aggregate Statistics

Computed across all judgments for a dataset:

| Statistic | Description |
|-----------|-------------|
| `mean` | Average disambiguation score across ambiguous questions. |
| `median` | Median disambiguation score. |
| `std` | Standard deviation of scores. |
| `min` | Lowest disambiguation score. |
| `max` | Highest disambiguation score. |
| `count` | Number of questions with disambiguation judgment (ambiguous only). |
| `total_interpretations` | Sum of all possible interpretations across questions. |
| `total_covered` | Sum of interpretations actually covered. |
| `coverage_rate` | Overall coverage: `total_covered / total_interpretations`. |

### Interpretation

- **High mean (> 0.7)**: RAG system addresses multiple interpretations well.
- **Low coverage_rate (< 0.5)**: Many interpretations are being missed.
- **coverage_rate = 1.0**: All interpretations covered for all questions (ideal for ambiguous questions).
- **High std**: Inconsistent at handling ambiguity — some questions well-handled, others not.

**Note**: `count` will be lower than `total_judged` since this judgment only applies to questions with multiple interpretations.

---

## Long-Form Judgment

### Purpose
Evaluates the quality of complete answer explanations, particularly important for datasets like **ASQA** where long-form explanations are expected.

### Metrics

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `quality_score` | Float | [0.0, 1.0] | Overall quality of the answer. Combines factuality, completeness, and coherence. |
| `factuality` | Float | [0.0, 1.0] | Accuracy of facts and claims made. 0 = contains false information, 1 = all verifiable facts correct. |
| `completeness` | Float | [0.0, 1.0] | Whether the answer addresses all aspects of the question. 0 = missing key info, 1 = thorough. |
| `coherence` | Float | [0.0, 1.0] | Clarity and logical flow of explanation. 0 = confusing/scattered, 1 = well-structured. |
| `details` | String | - | Specific feedback on strengths and weaknesses. |
| `reasoning` | String | - | Explanation of the quality judgment. |

### Aggregate Statistics

Computed across all judgments for a dataset:

| Statistic | Description |
|-----------|-------------|
| `mean_quality` | Average overall quality score. |
| `mean_factuality` | Average factuality across answers. |
| `mean_completeness` | Average completeness across answers. |
| `mean_coherence` | Average coherence across answers. |
| `count` | Number of long-form judgments (typically ASQA only). |

### Interpretation

- **mean_quality > 0.75**: Excellent overall answer quality.
- **mean_factuality < 0.6**: Generated answers contain factual errors or hallucinations.
- **mean_completeness < 0.5**: Answers miss important information from ground truth.
- **mean_coherence < 0.4**: Answers are confusing or poorly structured.
- **mean_quality > mean_factuality**: Answers are well-written but contain errors; focus on factuality improvement.

---

## Aggregate Metrics Summary

### What Gets Aggregated

When judging multiple results, the evaluator computes aggregate statistics across all judgments:

```json
{
  "aggregate_metrics": {
    "total_judged": 50,
    "similarity_scores": {
      "mean": 0.68,
      "median": 0.72,
      "std": 0.21,
      "min": 0.0,
      "max": 1.0,
      "count": 50
    },
    "disambiguation_scores": {
      "mean": 0.55,
      "median": 0.60,
      "std": 0.25,
      "min": 0.0,
      "max": 1.0,
      "count": 12,
      "total_interpretations": 28,
      "total_covered": 18,
      "coverage_rate": 0.64
    },
    "long_form_scores": {
      "mean_quality": 0.70,
      "mean_factuality": 0.72,
      "mean_completeness": 0.68,
      "mean_coherence": 0.71,
      "count": 50
    }
  }
}
```

### Interpretation

These aggregate metrics let you compare different RAG configurations:

| Comparison | What to Look For |
|-----------|------------------|
| **Different retrieval methods** (sparse vs dense vs hybrid) | Which has highest similarity_score mean? Consistency (lower std)? |
| **Different LLMs** (qwen vs gpt-4o-mini) | Which produces more factual answers? More complete? More coherent? |
| **Different datasets** (AmbigNQ vs ASQA) | How well does system handle ambiguity? Long-form quality? |
| **Different augmentation strategies** (vanilla vs agentic vs iterative) | Does agentic improve disambiguation? Does iterative improve factuality? |

---

## Data Structure Example

### Single Judgment Object

```json
{
  "question": "What is the capital of France?",
  "generated_answer": "The capital of France is Paris, located in the north-central part of the country on the Seine River.",
  "ground_truth_answers": ["Paris"],
  "dataset": "asqa",
  "similarity": {
    "similarity_score": 0.92,
    "is_similar": true,
    "matched_answer": "Paris",
    "coverage_score": 1.0,
    "reasoning": "The generated answer correctly identifies Paris as the capital."
  },
  "disambiguation": null,
  "long_form": {
    "quality_score": 0.85,
    "factuality": 0.9,
    "completeness": 0.8,
    "coherence": 0.85,
    "details": "Answer is accurate and clear. Could include more details about Paris.",
    "reasoning": "Well-structured explanation with accurate geographic information."
  }
}
```

---

## Comparison with Other Metrics

| Metric | Judgment Metrics | F1/ROUGE/BLEU |
|--------|-----------------|------------------|
| **Measures** | Semantic meaning and quality | Lexical overlap |
| **Sensitive to** | Paraphrase ability | Exact word matches |
| **Good for** | Ambiguous questions, meaning | Standard QA evaluation |
| **Bias** | LLM bias possible | Favors reference wording |
| **Computation** | Slower (LLM API calls) | Fast (string matching) |

Judge metrics are **complementary** to traditional metrics like F1, ROUGE, and BLEU.

---

## Practical Usage

### For Results.json Comparison

1. Run judge on all results files in `results/` directory
2. Compare `similarity_scores.mean` across different retrieval methods
3. Check `disambiguation_scores.coverage_rate` for ambiguous question performance
4. Look at `long_form_scores` for ASQA evaluation

### For Individual Result Analysis

1. Look at `is_similar` — is the answer semantically correct?
2. Check `interpretations_covered` — did we address all ambiguities?
3. Review `factuality` — are facts accurate?
4. Read `reasoning` — understand judge's rationale

### For System Improvement

- **Low similarity_score**: Improve retrieval or generation; check for hallucinations
- **Low coverage_rate**: Add multi-interpretation handling
- **Low factuality**: Improve retriever relevance; reduce hallucination
- **High std deviation**: System is inconsistent; debug failure cases

---

## Technical Notes

### Judge Model
- Default: `gpt-4o-mini` (fast, cost-effective)
- Can specify: `gpt-4-turbo`, `gpt-3.5-turbo`
- Uses cached judgments to reduce API costs

### Computation Time
- ~1-2 seconds per result (including API latency)
- Batch processing recommended to reduce overhead
- Caching enabled by default

### Limitations
- LLM judges can have biases
- May not perfectly capture nuanced meaning
- Edge cases (very long answers, unusual formats) may confuse judge
- Requires valid OpenAI API key
