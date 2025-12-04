"""
Judge all RAG results across datasets and compute aggregate metrics.

This script evaluates all results.json files and computes semantic judgments
along with aggregate statistics.

Usage:
    python scripts/judge_all_datasets.py --limit 50
    python scripts/judge_all_datasets.py --verbose
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
    elif isinstance(obj, (list, tuple)):
        return [convert_enums_to_values(item) for item in obj]
    return obj


class JudgeEvaluator:
    """Evaluates RAG results using an LLM judge."""
    
    def __init__(self, judge_model: str = "gpt-4o-mini", use_cache: bool = True):
        """Initialize judge evaluator."""
        self.base_judge = OpenAIJudge(model=judge_model)
        self.judge = CachedJudge(self.base_judge) if use_cache else self.base_judge
        self.judge_model = judge_model
        
    async def judge_result(
        self,
        result: Dict[str, Any],
        dataset: str = "ambignq"
    ) -> Dict[str, Any]:
        """Judge a single RAG result."""
        question = result.get("question", "")
        generated_answer = result.get("generated_answer", "")
        reference_data = result.get("reference_data", {})
        
        # Extract ground truth based on dataset
        if dataset == "asqa":
            ground_truth_answers = reference_data.get("all_short_answers", [])
            qa_pairs = reference_data.get("qa_pairs", [])
            long_answer_refs = [ann.get("long_answer", "") for ann in reference_data.get("annotations", [])]
        else:  # ambignq
            annotations = reference_data.get("annotations", []) or []
            ground_truth_answers = []
            qa_pairs = []

            # Some results use 'nq_answer' for short canonical answers
            if reference_data.get("nq_answer"):
                if isinstance(reference_data.get("nq_answer"), list):
                    ground_truth_answers.extend(reference_data.get("nq_answer", []))
                else:
                    ground_truth_answers.append(reference_data.get("nq_answer"))

            # Parse annotations: can be 'singleAnswer' with 'answer' list
            for ann in annotations:
                ann_type = ann.get("type", "").lower()
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
            seen = set()
            gt_unique = []
            for a in ground_truth_answers:
                if a not in seen:
                    seen.add(a)
                    gt_unique.append(a)
            ground_truth_answers = gt_unique

            long_answer_refs = []
        
        judgment = {
            "question": question,
            "generated_answer": generated_answer,
            "ground_truth_answers": ground_truth_answers,
            "qa_pairs": qa_pairs,
            "dataset": dataset
        }
        
        # Judge answer similarity to ground truth
        if ground_truth_answers:
            similarity_judgment = await self.judge.judge_answer_similarity(
                question=question,
                generated_answer=generated_answer,
                ground_truth_answers=ground_truth_answers
            )
            judgment["similarity"] = similarity_judgment
        
        # Judge disambiguation coverage if multiple interpretations
        if len(qa_pairs) > 1:
            disambiguation_judgment = await self.judge.judge_disambiguation(
                question=question,
                generated_answer=generated_answer,
                ground_truth_interpretations=qa_pairs,
                dataset=dataset
            )
            judgment["disambiguation"] = disambiguation_judgment
        
        # Judge long-form answer quality (ASQA)
        if dataset == "asqa" and long_answer_refs:
            long_form_judgment = await self.judge.judge_long_form_answer(
                question=question,
                generated_answer=generated_answer,
                reference_answer=long_answer_refs[0]
            )
            judgment["long_form"] = long_form_judgment
        
        return judgment
    
    async def judge_results(
        self,
        results: List[Dict[str, Any]],
        dataset: Optional[str] = None,
        limit: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Judge multiple RAG results."""
        if limit:
            results = results[:limit]
        
        judgments = []
        original_queries = []
        
        for i, result in enumerate(results):
            if (i + 1) % 10 == 0:
                logger.info(f"Judging result {i+1}/{len(results)}...")
            
            # Auto-detect dataset if needed
            if dataset is None:
                detected = detect_dataset_from_item(result.get("reference_data", {}))
                dataset = detected
            
            judgment = await self.judge_result(result, dataset=dataset)
            judgments.append(judgment)
            
            # Store original query data (use the judgment's ground truth extraction)
            original_queries.append({
                "question": judgment.get("question", ""),
                "ground_truth_answers": judgment.get("ground_truth_answers", []),
                "dataset": judgment.get("dataset", dataset)
            })
        
        return judgments, original_queries
    
    def compute_aggregate_scores(self, judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate scores across all judgments."""
        if not judgments:
            return {}
        
        metrics = {
            "total_judged": len(judgments),
            "similarity_scores": {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0
            },
            "disambiguation_scores": {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
                "total_interpretations": 0,
                "total_covered": 0,
                "coverage_rate": 0.0
            },
            "long_form_scores": {
                "mean_quality": 0.0,
                "mean_factuality": 0.0,
                "mean_completeness": 0.0,
                "mean_coherence": 0.0,
                "count": 0
            }
        }
        
        similarity_scores = []
        disambiguation_scores = []
        long_form_quality = []
        long_form_factuality = []
        long_form_completeness = []
        long_form_coherence = []
        
        for j in judgments:
            if "similarity" in j:
                metrics["similarity_scores"]["count"] += 1
                score = j["similarity"].get("similarity_score", 0)
                similarity_scores.append(score)
            
            if "disambiguation" in j:
                metrics["disambiguation_scores"]["count"] += 1
                score = j["disambiguation"].get("disambiguation_score", 0)
                disambiguation_scores.append(score)
                
                covered = j["disambiguation"].get("interpretations_covered", 0)
                total = j["disambiguation"].get("total_interpretations", 1)
                metrics["disambiguation_scores"]["total_covered"] += covered
                metrics["disambiguation_scores"]["total_interpretations"] += total
            
            if "long_form" in j:
                metrics["long_form_scores"]["count"] += 1
                long_form_quality.append(j["long_form"].get("quality_score", 0))
                long_form_factuality.append(j["long_form"].get("factuality", 0))
                long_form_completeness.append(j["long_form"].get("completeness", 0))
                long_form_coherence.append(j["long_form"].get("coherence", 0))
        
        # Compute statistics for similarity
        if similarity_scores:
            metrics["similarity_scores"]["mean"] = mean(similarity_scores)
            metrics["similarity_scores"]["median"] = median(similarity_scores)
            metrics["similarity_scores"]["min"] = min(similarity_scores)
            metrics["similarity_scores"]["max"] = max(similarity_scores)
            if len(similarity_scores) > 1:
                metrics["similarity_scores"]["std"] = stdev(similarity_scores)
        
        # Compute statistics for disambiguation
        if disambiguation_scores:
            metrics["disambiguation_scores"]["mean"] = mean(disambiguation_scores)
            metrics["disambiguation_scores"]["median"] = median(disambiguation_scores)
            metrics["disambiguation_scores"]["min"] = min(disambiguation_scores)
            metrics["disambiguation_scores"]["max"] = max(disambiguation_scores)
            if len(disambiguation_scores) > 1:
                metrics["disambiguation_scores"]["std"] = stdev(disambiguation_scores)
            
            if metrics["disambiguation_scores"]["total_interpretations"] > 0:
                metrics["disambiguation_scores"]["coverage_rate"] = (
                    metrics["disambiguation_scores"]["total_covered"] /
                    metrics["disambiguation_scores"]["total_interpretations"]
                )
        
        # Compute statistics for long-form
        if long_form_quality:
            metrics["long_form_scores"]["mean_quality"] = mean(long_form_quality)
            metrics["long_form_scores"]["mean_factuality"] = mean(long_form_factuality)
            metrics["long_form_scores"]["mean_completeness"] = mean(long_form_completeness)
            metrics["long_form_scores"]["mean_coherence"] = mean(long_form_coherence)
        
        return metrics


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Judge all RAG results using LLM")
    
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
    
    # Find all results.json files
    results_dir = Path(__file__).parent.parent / "results"
    results_files = sorted(results_dir.glob("**/results.json"))
    
    logger.info(f"Found {len(results_files)} results files")
    
    # Initialize evaluator
    evaluator = JudgeEvaluator(judge_model=args.judge_model, use_cache=True)
    
    all_results = {}
    
    for results_path in results_files:
        # Get relative path for output
        rel_path = results_path.relative_to(results_dir)
        output_name = "_".join(rel_path.parts[:-1]) + "_judgments.json"
        output_path = Path(__file__).parent.parent / output_name
        
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
        
        if metrics["long_form_scores"]["count"] > 0:
            logger.info(f"\nLong-Form Answer Quality ({metrics['long_form_scores']['count']} items):")
            logger.info(f"  Mean Quality: {metrics['long_form_scores']['mean_quality']:.3f}")
            logger.info(f"  Mean Factuality: {metrics['long_form_scores']['mean_factuality']:.3f}")
            logger.info(f"  Mean Completeness: {metrics['long_form_scores']['mean_completeness']:.3f}")
            logger.info(f"  Mean Coherence: {metrics['long_form_scores']['mean_coherence']:.3f}")
        
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
    logger.info(f"\n{'='*70}\nSummary Across All Datasets\n{'='*70}")
    for dataset, metrics in all_results.items():
        logger.info(f"\n{dataset}:")
        logger.info(f"  Similarity (mean): {metrics['similarity_scores']['mean']:.3f}")
        logger.info(f"  Disambiguation (mean): {metrics['disambiguation_scores']['mean']:.3f}")
        if metrics["long_form_scores"]["count"] > 0:
            logger.info(f"  Long-form quality (mean): {metrics['long_form_scores']['mean_quality']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
