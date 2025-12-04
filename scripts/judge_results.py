"""
Judge-Based Evaluation Script

Demonstrates how to use the LLM judge to evaluate RAG results.
Compares generated answers against ground truth using semantic judgment.

Usage:
    python scripts/judge_results.py --results-path results/agentic/hybrid/gpt-4o-mini/results.json
    python scripts/judge_results.py --results-path results/agentic/hybrid/gpt-4o-mini/results.json --limit 10 --verbose
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from enum import Enum
from pathlib import Path
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
        """
        Initialize judge evaluator.
        
        Args:
            judge_model: OpenAI model to use for judging
            use_cache: Whether to cache judge results
        """
        self.base_judge = OpenAIJudge(model=judge_model)
        self.judge = CachedJudge(self.base_judge) if use_cache else self.base_judge
        self.judge_model = judge_model
        
    async def judge_result(
        self,
        result: Dict[str, Any],
        dataset: str = "ambignq"
    ) -> Dict[str, Any]:
        """
        Judge a single RAG result.
        
        Args:
            result: Result object with question, generated_answer, reference_data
            dataset: "ambignq" or "asqa"
            
        Returns:
            Judgment object with scores and reasoning
        """
        question = result.get("question", "")
        generated_answer = result.get("generated_answer", "")
        reference_data = result.get("reference_data", {})
        
        # Extract ground truth based on dataset
        if dataset == "asqa":
            ground_truth_answers = reference_data.get("all_short_answers", [])
            qa_pairs = reference_data.get("qa_pairs", [])
            long_answer_refs = [ann.get("long_answer", "") for ann in reference_data.get("annotations", [])]
        else:  # ambignq
            ground_truth_answers = reference_data.get("all_short_answers", [])
            annotations = reference_data.get("annotations", [])
            # For ambignq, extract answer sets from annotations
            qa_pairs = []
            if annotations:
                if "multipleQAs" in annotations[0]:
                    qa_pairs = [
                        {
                            "question": qa.get("question", ""),
                            "short_answers": qa.get("answer", [])
                        }
                        for qa in annotations[0].get("qaPairs", [])
                    ]
                else:
                    # Single answer case
                    answer = annotations[0].get("answer", [])
                    if answer:
                        qa_pairs = [{"question": question, "short_answers": answer}]
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
            logger.info(f"  Judging answer similarity...")
            similarity_judgment = await self.judge.judge_answer_similarity(
                question=question,
                generated_answer=generated_answer,
                ground_truth_answers=ground_truth_answers
            )
            judgment["similarity"] = similarity_judgment
            logger.info(f"    Similarity: {similarity_judgment.get('similarity_score', 0):.2f} - {similarity_judgment.get('reasoning', '')[:100]}")
        
        # Judge disambiguation coverage if multiple interpretations
        if len(qa_pairs) > 1:
            logger.info(f"  Judging disambiguation coverage ({len(qa_pairs)} interpretations)...")
            disambiguation_judgment = await self.judge.judge_disambiguation(
                question=question,
                generated_answer=generated_answer,
                ground_truth_interpretations=qa_pairs,
                dataset=dataset
            )
            judgment["disambiguation"] = disambiguation_judgment
            logger.info(f"    Coverage: {disambiguation_judgment.get('interpretations_covered', 0)}/{len(qa_pairs)} interpretations")
        
        # Judge long-form answer quality (ASQA)
        if dataset == "asqa" and long_answer_refs:
            logger.info(f"  Judging long-form answer quality...")
            long_form_judgment = await self.judge.judge_long_form_answer(
                question=question,
                generated_answer=generated_answer,
                reference_answer=long_answer_refs[0]
            )
            judgment["long_form"] = long_form_judgment
            logger.info(f"    Quality: {long_form_judgment.get('quality_score', 0):.2f}")
        
        return judgment
    
    async def judge_results(
        self,
        results: List[Dict[str, Any]],
        dataset: Optional[str] = None,
        limit: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Judge multiple RAG results.
        
        Args:
            results: List of result objects
            dataset: Dataset type (auto-detected if None)
            limit: Max results to judge
            verbose: Print detailed output
            
        Returns:
            Tuple of (judgments list, original_queries list)
        """
        if limit:
            results = results[:limit]
        
        judgments = []
        original_queries = []
        
        for i, result in enumerate(results):
            logger.info(f"\nJudging result {i+1}/{len(results)}...")
            
            # Auto-detect dataset if needed
            if dataset is None:
                detected = detect_dataset_from_item(result.get("reference_data", {}))
                dataset = detected
            
            judgment = await self.judge_result(result, dataset=dataset)
            judgments.append(judgment)
            
            # Store original query data
            question = result.get("question", "")
            reference_data = result.get("reference_data", {})
            ground_truth_answers = reference_data.get("all_short_answers", [])
            
            original_queries.append({
                "question": question,
                "ground_truth_answers": ground_truth_answers,
                "dataset": dataset
            })
            
            if verbose:
                logger.info(f"\nJudgment for question: {judgment.get('question', '')[:100]}")
                if "similarity" in judgment:
                    logger.info(f"  Similarity Score: {judgment['similarity'].get('similarity_score', 0):.2f}")
                if "disambiguation" in judgment:
                    logger.info(f"  Disambiguation Score: {judgment['disambiguation'].get('disambiguation_score', 0):.2f}")
                if "long_form" in judgment:
                    logger.info(f"  Long-form Quality: {judgment['long_form'].get('quality_score', 0):.2f}")
        
        return judgments, original_queries
    
    def compute_aggregate_scores(self, judgments: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute aggregate scores across all judgments.
        
        Args:
            judgments: List of judgment objects
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not judgments:
            return {}
        
        metrics = {
            "total_judged": len(judgments),
            "average_similarity_score": 0.0,
            "average_disambiguation_score": 0.0,
            "average_long_form_quality": 0.0,
            "num_with_similarity": 0,
            "num_with_disambiguation": 0,
            "num_with_long_form": 0,
        }
        
        similarity_scores = []
        disambiguation_scores = []
        long_form_scores = []
        
        for j in judgments:
            if "similarity" in j:
                metrics["num_with_similarity"] += 1
                score = j["similarity"].get("similarity_score", 0)
                similarity_scores.append(score)
            
            if "disambiguation" in j:
                metrics["num_with_disambiguation"] += 1
                score = j["disambiguation"].get("disambiguation_score", 0)
                disambiguation_scores.append(score)
            
            if "long_form" in j:
                metrics["num_with_long_form"] += 1
                score = j["long_form"].get("quality_score", 0)
                long_form_scores.append(score)
        
        if similarity_scores:
            metrics["average_similarity_score"] = sum(similarity_scores) / len(similarity_scores)
        
        if disambiguation_scores:
            metrics["average_disambiguation_score"] = sum(disambiguation_scores) / len(disambiguation_scores)
        
        if long_form_scores:
            metrics["average_long_form_quality"] = sum(long_form_scores) / len(long_form_scores)
        
        return metrics


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Judge RAG results using LLM")
    
    parser.add_argument(
        "--results-path",
        type=str,
        required=True,
        help="Path to results JSON file"
    )
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
        help="Max results to judge"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--save-judgments",
        type=str,
        default=None,
        help="Save judgments to file"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Load results
    logger.info(f"Loading results from {args.results_path}...")
    with open(args.results_path, 'r') as f:
        data = json.load(f)
    
    results = data.get("results", [])
    logger.info(f"Loaded {len(results)} results")
    
    # Initialize evaluator
    evaluator = JudgeEvaluator(judge_model=args.judge_model, use_cache=True)
    
    # Judge results
    logger.info(f"\nStarting judging with {args.judge_model}...")
    judgments, original_queries = await evaluator.judge_results(
        results,
        limit=args.limit,
        verbose=args.verbose
    )
    
    # Compute aggregates
    logger.info(f"\n{'='*60}\nAggregate Judge Scores\n{'='*60}")
    metrics = evaluator.compute_aggregate_scores(judgments)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.3f}")
        else:
            logger.info(f"{key}: {value}")
    
    # Save judgments if requested
    if args.save_judgments:
        output_data = {
            "judge_model": args.judge_model,
            "judgments": judgments,
            "original_queries": original_queries,
            "aggregate_metrics": metrics
        }
        
        # Convert any Enum objects to their string values
        output_data = convert_enums_to_values(output_data)
        
        with open(args.save_judgments, 'w') as f:
            json.dump(output_data, f, indent=2, cls=EnumEncoder)
        
        logger.info(f"\nJudgments saved to {args.save_judgments}")


if __name__ == "__main__":
    asyncio.run(main())
