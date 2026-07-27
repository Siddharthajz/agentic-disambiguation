# LLM Judge Integration Guide

This document explains how to use the LLM judge module to evaluate generated answers against ground truth.

## Overview

The judge module provides an **OpenAI-based semantic judge** that evaluates:

1. **Answer Similarity**: Whether generated answers are semantically similar to ground truth
2. **Disambiguation Coverage**: Which interpretations of ambiguous questions are addressed
3. **Long-Form Quality**: Quality of long-form answers (for ASQA dataset)

## Components

### `judge.py` - Core Judge Module

Located at: `src/core/judge.py`

#### `BaseJudge` (Abstract)
Interface for all judges. Implement for custom judge implementations.

#### `OpenAIJudge`
LLM-based judge using OpenAI API with structured JSON output.

```python
from src.core import OpenAIJudge

judge = OpenAIJudge(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key="your-api-key"
)
```

#### `CachedJudge`
Wrapper that adds caching to avoid redundant API calls for identical judgments.

```python
from src.core import OpenAIJudge, CachedJudge

base_judge = OpenAIJudge(model="gpt-4o-mini")
judge = CachedJudge(base_judge)
```

## Usage Examples

### 1. Judge Answer Similarity

Determine if a generated answer is semantically similar to ground truth answers:

```python
import asyncio
from src.core import OpenAIJudge

async def judge_answer():
    judge = OpenAIJudge(model="gpt-4o-mini")
    
    result = await judge.judge_answer_similarity(
        question="What is the capital of France?",
        generated_answer="The capital of France is Paris.",
        ground_truth_answers=["Paris", "The capital is Paris"]
    )
    
    print(f"Similarity Score: {result['similarity_score']}")
    print(f"Is Similar: {result['is_similar']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Coverage Score: {result['coverage_score']}")
    
    # Result structure:
    # {
    #     "similarity_score": 0.95,
    #     "is_similar": true,
    #     "matched_answer": "Paris",
    #     "reasoning": "The generated answer directly matches...",
    #     "coverage_score": 1.0,
    #     "coverage_reasoning": "All key facts covered",
    #     "generation_time": 0.5,
    #     "total_tokens": 150
    # }

asyncio.run(judge_answer())
```

### 2. Judge Disambiguation Coverage

Evaluate if a generated answer addresses all interpretations of an ambiguous question:

#### For AmbigNQ:
```python
async def judge_ambignq():
    judge = OpenAIJudge(model="gpt-4o-mini")
    
    ground_truth = [
        ["1979"],           # Interpretation 1: NBA
        ["1967"]            # Interpretation 2: ABA
    ]
    
    result = await judge.judge_disambiguation(
        question="When was the 3-point line introduced?",
        generated_answer="The 3-point line was introduced to the NBA in 1979, though the ABA had it since 1967.",
        ground_truth_interpretations=ground_truth,
        dataset="ambignq"
    )
    
    print(f"Disambiguation Score: {result['disambiguation_score']}")
    print(f"Covered: {result['interpretations_covered']}/{result['total_interpretations']}")

asyncio.run(judge_ambignq())
```

#### For ASQA:
```python
async def judge_asqa():
    judge = OpenAIJudge(model="gpt-4o-mini")
    
    ground_truth = [
        {
            "question": "When was the NBA 3-point line introduced?",
            "short_answers": ["1979", "1979-80 season"]
        },
        {
            "question": "When was the ABA 3-point line introduced?",
            "short_answers": ["1967", "1967-68 season"]
        }
    ]
    
    result = await judge.judge_disambiguation(
        question="When was the 3-point line introduced?",
        generated_answer="The 3-point line was introduced to the NBA in 1979...",
        ground_truth_interpretations=ground_truth,
        dataset="asqa"
    )
    
    print(f"Covered interpretations: {result['covered_interpretations']}")
    print(f"Missing: {result['missing_interpretations']}")

asyncio.run(judge_asqa())
```

### 3. Judge Long-Form Answer Quality (ASQA)

Evaluate long-form answer against reference:

```python
async def judge_long_form():
    judge = OpenAIJudge(model="gpt-4o-mini")
    
    result = await judge.judge_long_form_answer(
        question="When was the 3-point line introduced?",
        generated_answer="The 3-point line was introduced to the NBA in 1979 for the 1979-80 season...",
        reference_answer="The NBA adopted the three-point line for the 1979-80 season, borrowing from the ABA which had used it since 1967..."
    )
    
    print(f"Quality Score: {result['quality_score']}")
    print(f"Factuality: {result['factuality']}")
    print(f"Completeness: {result['completeness']}")
    print(f"Coherence: {result['coherence']}")
    print(f"Strengths: {result['key_strengths']}")
    print(f"Weaknesses: {result['key_weaknesses']}")

asyncio.run(judge_long_form())
```

## Integration with RAG Pipeline

### Option 1: Add Judge Node to LangGraph

In `src/agentic_disambiguation.py`:

```python
from src.core import CachedJudge, OpenAIJudge

class LangGraphAgenticDisambiguation:
    def __init__(self, config, dataset="ambignq"):
        # ... existing initialization ...
        
        if config.use_judge:
            base_judge = OpenAIJudge(
                model=config.judge_model,
                api_key=config.openai_api_key
            )
            self.judge = CachedJudge(base_judge)
        else:
            self.judge = None
    
    async def judge_answer_node(self, state: AgentState) -> AgentState:
        """LangGraph node for judging answer quality."""
        if not self.judge:
            return state
        
        try:
            ground_truth = state["reference_data"].get("all_short_answers", [])
            
            judgment = await self.judge.judge_answer_similarity(
                question=state["question"],
                generated_answer=state["generated_answer"],
                ground_truth_answers=ground_truth
            )
            
            if not state.get("metadata"):
                state["metadata"] = {}
            
            state["metadata"]["judge_judgment"] = judgment
            
        except Exception as e:
            logger.error(f"Judging failed: {e}")
        
        return state
```

Then add to workflow:
```python
def _build_workflow(self):
    workflow = StateGraph(AgentState)
    
    # Add judge node
    workflow.add_node("judge_answer", self.judge_answer_node)
    
    # Add edge before final output
    workflow.add_edge("synthesize_answer", "judge_answer")
    workflow.add_edge("judge_answer", END)
    
    return workflow.compile()
```

### Option 2: Batch Post-Processing

Judge all results after generation completes:

```python
from scripts.judge_results import JudgeEvaluator

async def evaluate_with_judge(results, dataset="ambignq"):
    evaluator = JudgeEvaluator(judge_model="gpt-4o-mini", use_cache=True)
    judgments = await evaluator.judge_results(results, dataset=dataset)
    metrics = evaluator.compute_aggregate_scores(judgments)
    return judgments, metrics
```

## Command-Line Usage

### Judge existing results:

```bash
# Judge a specific results file
python scripts/judge_results.py \
  --results-path results/agentic/hybrid/gpt-4o-mini/results.json

# Limit to 10 examples with verbose output
python scripts/judge_results.py \
  --results-path results/agentic/hybrid/gpt-4o-mini/results.json \
  --limit 10 \
  --verbose

# Save judgments to file
python scripts/judge_results.py \
  --results-path results/agentic/hybrid/gpt-4o-mini/results.json \
  --save-judgments judgments.json

# Use custom judge model
python scripts/judge_results.py \
  --results-path results/agentic/hybrid/gpt-4o-mini/results.json \
  --judge-model gpt-4o
```

## Configuration

Add to `src/core/config.py`:

```python
@dataclass
class RAGConfig:
    # ... existing fields ...
    
    # Judge configuration
    use_judge: bool = False
    judge_model: str = "gpt-4o-mini"
    judge_cache_size: int = 1000
```

Add CLI arguments to `src/agentic_disambiguation.py`:

```python
parser.add_argument("--use-judge", action="store_true", help="Enable LLM judge")
parser.add_argument("--judge-model", default="gpt-4o-mini", help="Judge model")
```

## Output Format

Judgments are stored in results as:

```json
{
  "metadata": {
    "judge_judgment": {
      "similarity_score": 0.85,
      "is_similar": true,
      "matched_answer": "...",
      "reasoning": "...",
      "coverage_score": 0.9,
      "disambiguation": {
        "disambiguation_score": 0.75,
        "interpretations_covered": 2,
        "covered_interpretations": [0, 1],
        "missing_interpretations": [2]
      },
      "long_form": {
        "quality_score": 0.88,
        "factuality": 0.9,
        "completeness": 0.85,
        "coherence": 0.88
      }
    }
  }
}
```

## Caching

The `CachedJudge` wrapper caches results to avoid redundant API calls:

```python
from src.core import OpenAIJudge, CachedJudge

judge = CachedJudge(OpenAIJudge(model="gpt-4o-mini"))

# First call hits API
result1 = await judge.judge_answer_similarity(...)

# Second call with same inputs returns cached result
result2 = await judge.judge_answer_similarity(...)

# Check cache stats
stats = judge.get_stats()
print(f"Cached judgments: {stats['cached_judgments']}")
```

## Cost Considerations

- Each judgment call costs ~$0.0002-0.0005 depending on model and response length
- For 1000 examples with 3 judgment calls each: ~$0.60-1.50
- Use caching to reduce costs when judging similar answer patterns
- Consider using cheaper models (gpt-4o-mini) for initial evaluation

## Troubleshooting

**Issue**: `OPENAI_API_KEY` not set
```bash
export OPENAI_API_KEY="your-key"
# or add to .env file
```

**Issue**: Rate limiting
- Add delay between requests: `await asyncio.sleep(0.5)`
- Reduce concurrency

**Issue**: JSON parsing errors
- Check that OpenAI is returning valid JSON
- Verify `response_format={"type": "json_object"}` is supported by model

## Future Enhancements

1. **Multiple Judge Models**: Support Claude, Llama, etc.
2. **Few-shot Examples**: Provide examples to improve judgment quality
3. **Ensemble Judges**: Combine multiple judges for more robust scoring
4. **Fine-tuned Judge**: Train judge on labeled comparison data
5. **Human-in-the-Loop**: Integrate human feedback to calibrate judge
