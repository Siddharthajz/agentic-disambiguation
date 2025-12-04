"""
Judge AmbigNQ results with the fixed disambiguation logic.

This script only processes AmbigNQ datasets and saves results to a separate
folder (ambignq_fixed_judgments) to preserve the original results.

Usage:
    python scripts/judge_ambignq_fixed.py --limit 50
    python scripts/judge_ambignq_fixed.py --verbose
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from enum import Enum
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import OpenAIJudge, CachedJudge, detect_dataset_from_item
from src.evaluation import RAGEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnumEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Enum values."""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def convert_enums_to_values(obj: Any) -> Any:
    """Recursively convert Enum objects to their values."""
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {k: convert_enums_to_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_enums_to_values(item) for item in obj]
    return obj


class JudgeEvaluator:
    """Evaluates RAG results using an LLM judge."""

    def __init__(self, judge_model: str = "gpt-4o-mini", use_cache: bool = True):
        self.judge_model = judge_model
        self.base_judge = OpenAIJudge(model=judge_model)
        self.judge = CachedJudge(self.base_judge) if use_cache else self.base_judge

    async def judge_result(
        self,
        result: Dict[str, Any],
        reference_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Judge a single result."""
        question = result.get("question", "")
        generated_answer = result.get("generated_answer", "")
        dataset = "ambignq"

        # Extract ground truth from reference_data (same logic as judge_all_datasets.py)
        annotations = reference_data.get("annotations", []) or []
        ground_truth_answers = []
        qa_pairs = []

        # Some results use 'nq_answer' for short canonical answers
        if reference_data.get("nq_answer"):
            if isinstance(reference_data.get("nq_answer"), list):
                ground_truth_answers.extend(reference_data.get("nq_answer", []))
            else:
                ground_truth_answers.append(reference_data.get("nq_answer"))

        # Parse annotations: can be 'singleAnswer' with 'answer' list or 'multipleQAs' with qaPairs
        for ann in annotations:
            # multipleQAs -> has qaPairs
            if "qaPairs" in ann:
                for qa in ann.get("qaPairs", []):
                    qa_q = qa.get("question", "")
                    qa_a = qa.get("answer", []) or []
                    qa_pairs.append({"question": qa_q or question, "short_answers": qa_a})
                    ground_truth_answers.extend(qa_a)
            else:
                # single/multi answer formats
                ans = ann.get("answer") or ann.get("answers") or []
                if isinstance(ans, list) and ans:
                    ground_truth_answers.extend(ans)
                    qa_pairs.append({"question": question, "short_answers": ans})

        # Deduplicate ground truth answers
        ground_truth_answers = list(dict.fromkeys(ground_truth_answers))

        # Similarity judgment
        similarity_judgment = await self.judge.judge_answer_similarity(
            question=question,
            generated_answer=generated_answer,
            ground_truth_answers=ground_truth_answers
        )

        # Disambiguation judgment (for questions with multiple interpretations)
        disambiguation_judgment = None
        if len(qa_pairs) > 1:
            disambiguation_judgment = await self.judge.judge_disambiguation(
                question=question,
                generated_answer=generated_answer,
                ground_truth_interpretations=qa_pairs,
                dataset=dataset
            )

        judgment = {
            "question": question,
            "generated_answer": generated_answer,
            "ground_truth_answers": ground_truth_answers,
            "qa_pairs": qa_pairs,
            "dataset": dataset,
            "similarity": similarity_judgment,
            "disambiguation": disambiguation_judgment,
        }

        original_query = {
            "question": question,
            "ground_truth_answers": ground_truth_answers,
            "qa_pairs": qa_pairs
        }

        return judgment, original_query

    async def judge_results(
        self,
        results: List[Dict[str, Any]],
        limit: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Judge a batch of results."""
        judgments = []
        original_queries = []

        # Limit if specified
        if limit:
            results = results[:limit]

        for i, result in enumerate(results):
            if verbose and i % 50 == 0:
                logger.info(f"Processing result {i+1}/{len(results)}")

            # Extract reference data from the result itself
            reference_data = result.get("reference_data", {})

            if not reference_data:
                logger.warning(f"No reference data for result {i}")
                continue

            try:
                judgment, original_query = await self.judge_result(result, reference_data)
                judgments.append(judgment)
                original_queries.append(original_query)
            except Exception as e:
                logger.error(f"Error judging result {i}: {e}")
                continue

        return judgments, original_queries

    def compute_aggregate_scores(self, judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate metrics from judgments."""
        similarity_scores = []
        disambiguation_scores = []
        long_form_scores = []

        total_interpretations = 0
        total_covered = 0

        for judgment in judgments:
            if judgment.get("similarity"):
                similarity_scores.append(judgment["similarity"].get("similarity_score", 0.0))

            if judgment.get("disambiguation"):
                dis = judgment["disambiguation"]
                disambiguation_scores.append(dis.get("disambiguation_score", 0.0))
                total_interpretations += dis.get("total_interpretations", 0)
                total_covered += dis.get("interpretations_covered", 0)

        # Compute statistics
        def compute_stats(scores):
            if not scores:
                return {
                    "count": 0,
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0
                }
            return {
                "count": len(scores),
                "mean": mean(scores),
                "median": median(scores),
                "std": stdev(scores) if len(scores) > 1 else 0.0,
                "min": min(scores),
                "max": max(scores)
            }

        return {
            "total_judged": len(judgments),
            "similarity_scores": compute_stats(similarity_scores),
            "disambiguation_scores": {
                **compute_stats(disambiguation_scores),
                "total_interpretations": total_interpretations,
                "total_covered": total_covered,
                "coverage_rate": total_covered / max(1, total_interpretations)
            },
            "long_form_scores": {"count": 0}  # AmbigNQ doesn't have long-form
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Judge AmbigNQ results using LLM with fixed disambiguation")
    
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max results to judge per dataset"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "ambignq_fixed_judgments"
    output_dir.mkdir(exist_ok=True)
    
    # Find all AmbigNQ results.json files
    results_dir = Path(__file__).parent.parent / "results"
    results_files = sorted(results_dir.glob("ambignq/**/results.json"))
    
    logger.info(f"Found {len(results_files)} AmbigNQ results files")
    
    # Initialize evaluator
    evaluator = JudgeEvaluator(judge_model=args.judge_model, use_cache=True)
    
    all_results = {}
    
    for results_path in results_files:
        # Get relative path for output
        rel_path = results_path.relative_to(results_dir)
        output_name = "_".join(rel_path.parts[:-1]) + "_judgments.json"
        output_path = output_dir / output_name
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {rel_path}")
        logger.info(f"Output: {output_name}")
        logger.info(f"{'='*70}")
        
        # Load results
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {results_path}: {e}")
            continue
        
        results = data.get("results", [])
        if not results:
            logger.warning(f"No results found in {results_path}")
            continue
        
        logger.info(f"Loaded {len(results)} results")
        
        # Judge results
        logger.info(f"Starting judging with {args.judge_model}...")
        judgments, original_queries = await evaluator.judge_results(
            results,
            limit=args.limit,
            verbose=args.verbose
        )
        
        # Compute aggregates
        logger.info(f"Computing aggregate metrics...")
        metrics = evaluator.compute_aggregate_scores(judgments)
        
        logger.info(f"\n{'='*70}\nJudge Metrics for {rel_path}\n{'='*70}")
        logger.info(f"Total Judged: {metrics['total_judged']}")
        
        if metrics["similarity_scores"]["count"] > 0:
            logger.info(f"\nSimilarity Judgment ({metrics['similarity_scores']['count']} items):")
            logger.info(f"  Mean: {metrics['similarity_scores']['mean']:.3f}")
            logger.info(f"  Median: {metrics['similarity_scores']['median']:.3f}")
            logger.info(f"  Std Dev: {metrics['similarity_scores']['std']:.3f}")
            logger.info(f"  Range: [{metrics['similarity_scores']['min']:.3f}, {metrics['similarity_scores']['max']:.3f}]")
        
        if metrics["disambiguation_scores"]["count"] > 0:
            logger.info(f"\nDisambiguation Coverage ({metrics['disambiguation_scores']['count']} items):")
            logger.info(f"  Mean Score: {metrics['disambiguation_scores']['mean']:.3f}")
            logger.info(f"  Median Score: {metrics['disambiguation_scores']['median']:.3f}")
            logger.info(f"  Overall Coverage Rate: {metrics['disambiguation_scores']['coverage_rate']:.3f}")
            logger.info(f"  Total Interpretations Covered: {metrics['disambiguation_scores']['total_covered']}/{metrics['disambiguation_scores']['total_interpretations']}")
        
        # Save judgments
        output_data = {
            "judge_model": args.judge_model,
            "results_path": str(rel_path),
            "judgments": judgments,
            "original_queries": original_queries,
            "aggregate_metrics": metrics
        }
        
        # Convert any Enum objects to their string values
        output_data = convert_enums_to_values(output_data)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, cls=EnumEncoder)
        
        logger.info(f"\nJudgments saved to {output_path}")
        all_results[str(rel_path)] = metrics
    
    # Save summary
    logger.info(f"\n{'='*70}\nSummary Across All AmbigNQ Datasets\n{'='*70}")
    for dataset, metrics in all_results.items():
        logger.info(f"\n{dataset}:")
        logger.info(f"  Similarity (mean): {metrics['similarity_scores']['mean']:.3f}")
        logger.info(f"  Disambiguation (mean): {metrics['disambiguation_scores']['mean']:.3f}")
        logger.info(f"  Coverage Rate: {metrics['disambiguation_scores']['coverage_rate']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
