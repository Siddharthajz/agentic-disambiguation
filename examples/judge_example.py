#!/usr/bin/env python3
"""
Quick Example: Using the LLM Judge

This script demonstrates how to use the judge to compare generated answers
against ground truth answers.

Usage:
    python examples/judge_example.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.core import OpenAIJudge, CachedJudge


async def example_1_answer_similarity():
    """Example 1: Judge if answer is similar to ground truth."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Answer Similarity Judgment")
    print("="*70)
    
    judge = CachedJudge(OpenAIJudge(model="gpt-4o-mini"))
    
    question = "When was the US break away from England?"
    generated = "The United States declared independence from England on July 4, 1776."
    ground_truth = [
        "July 4, 1776",
        "1776",
        "Declaration of Independence"
    ]
    
    print(f"\nQuestion: {question}")
    print(f"Generated Answer: {generated}")
    print(f"Ground Truth Options: {ground_truth}")
    
    result = await judge.judge_answer_similarity(
        question=question,
        generated_answer=generated,
        ground_truth_answers=ground_truth
    )
    
    print(f"\n--- Judge Output ---")
    print(f"Similarity Score: {result['similarity_score']:.2f}")
    print(f"Is Similar: {result['is_similar']}")
    print(f"Matched Answer: {result.get('matched_answer')}")
    print(f"Coverage Score: {result['coverage_score']:.2f}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Time: {result['generation_time']:.2f}s | Tokens: {result['total_tokens']}")


async def example_2_ambiguous_question():
    """Example 2: Judge disambiguation coverage."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Disambiguation Coverage (AmbigNQ)")
    print("="*70)
    
    judge = CachedJudge(OpenAIJudge(model="gpt-4o-mini"))
    
    question = "When was the NBA 3-point line introduced?"
    generated = (
        "The 3-point line was introduced to the NBA in the 1979-80 season. "
        "However, it originated earlier in the ABA (American Basketball Association) in 1967."
    )
    ground_truth = [
        ["1979", "1979-80 season"],  # Interpretation 1: NBA
        ["1967", "1967-68 season"]   # Interpretation 2: ABA
    ]
    
    print(f"\nQuestion: {question}")
    print(f"Generated Answer: {generated[:100]}...")
    print(f"Interpretations:")
    for i, answers in enumerate(ground_truth):
        print(f"  {i+1}. Valid answers: {', '.join(answers)}")
    
    result = await judge.judge_disambiguation(
        question=question,
        generated_answer=generated,
        ground_truth_interpretations=ground_truth,
        dataset="ambignq"
    )
    
    print(f"\n--- Judge Output ---")
    print(f"Disambiguation Score: {result['disambiguation_score']:.2f}")
    print(f"Interpretations Covered: {result['interpretations_covered']}/{result['total_interpretations']}")
    print(f"Covered: {result['covered_interpretations']}")
    print(f"Missing: {result['missing_interpretations']}")
    print(f"Reasoning: {result['reasoning'][:200]}...")
    print(f"Time: {result['generation_time']:.2f}s | Tokens: {result['total_tokens']}")


async def example_3_asqa_disambiguation():
    """Example 3: Judge ASQA disambiguation (with sub-questions)."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Disambiguation Coverage (ASQA)")
    print("="*70)
    
    judge = CachedJudge(OpenAIJudge(model="gpt-4o-mini"))
    
    question = "When was the 3-point line introduced?"
    generated = (
        "The 3-point line has a complex history in basketball. "
        "The NBA introduced it in 1979, while the ABA had adopted it in 1967. "
        "The international game (FIBA) also uses a 3-point line."
    )
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
    
    print(f"\nQuestion: {question}")
    print(f"Generated Answer: {generated[:120]}...")
    print(f"Interpretations (ASQA format):")
    for i, qa in enumerate(ground_truth):
        print(f"  {i+1}. \"{qa['question']}\"")
        print(f"     Valid answers: {', '.join(qa['short_answers'])}")
    
    result = await judge.judge_disambiguation(
        question=question,
        generated_answer=generated,
        ground_truth_interpretations=ground_truth,
        dataset="asqa"
    )
    
    print(f"\n--- Judge Output ---")
    print(f"Disambiguation Score: {result['disambiguation_score']:.2f}")
    print(f"Interpretations Covered: {result['interpretations_covered']}/{result['total_interpretations']}")
    print(f"Covered: {result['covered_interpretations']}")
    print(f"Missing: {result['missing_interpretations']}")
    print(f"Time: {result['generation_time']:.2f}s | Tokens: {result['total_tokens']}")


async def example_4_long_form():
    """Example 4: Judge long-form answer quality."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Long-Form Answer Quality (ASQA)")
    print("="*70)
    
    judge = CachedJudge(OpenAIJudge(model="gpt-4o-mini"))
    
    question = "When was the 3-point line introduced in basketball?"
    generated = (
        "The three-point line was introduced to basketball at different times "
        "in different leagues. The NBA adopted it for the 1979-80 season, "
        "bringing it from the American Basketball Association (ABA) which had used "
        "it since 1967-68."
    )
    reference = (
        "The three-point line has a rich history in basketball. The American "
        "Basketball Association (ABA) was the first to introduce the three-point "
        "line in the 1967-68 season. When the ABA and NBA merged in 1976, the NBA "
        "initially didn't use it, but adopted the three-point line for the 1979-80 "
        "season. This distance was eventually standardized."
    )
    
    print(f"\nQuestion: {question}")
    print(f"Generated Answer: {generated[:120]}...")
    print(f"Reference Answer: {reference[:120]}...")
    
    result = await judge.judge_long_form_answer(
        question=question,
        generated_answer=generated,
        reference_answer=reference
    )
    
    print(f"\n--- Judge Output ---")
    print(f"Quality Score: {result['quality_score']:.2f}")
    print(f"  - Factuality: {result['factuality']:.2f}")
    print(f"  - Completeness: {result['completeness']:.2f}")
    print(f"  - Coherence: {result['coherence']:.2f}")
    print(f"Strengths: {', '.join(result.get('key_strengths', []))}")
    print(f"Weaknesses: {', '.join(result.get('key_weaknesses', []))}")
    print(f"Reasoning: {result['reasoning'][:200]}...")
    print(f"Time: {result['generation_time']:.2f}s | Tokens: {result['total_tokens']}")


async def example_5_caching():
    """Example 5: Demonstrate caching benefits."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Judge Caching (API Cost Savings)")
    print("="*70)
    
    judge = CachedJudge(OpenAIJudge(model="gpt-4o-mini"))
    
    question = "What is the capital of France?"
    generated = "The capital of France is Paris."
    ground_truth = ["Paris", "The city of Paris"]
    
    print(f"\nFirst judgment call (hits API)...")
    result1 = await judge.judge_answer_similarity(question, generated, ground_truth)
    print(f"✓ Complete. Score: {result1['similarity_score']:.2f}")
    
    print(f"\nSecond judgment call with same inputs (cached)...")
    result2 = await judge.judge_answer_similarity(question, generated, ground_truth)
    print(f"✓ Complete (from cache). Score: {result2['similarity_score']:.2f}")
    
    stats = judge.get_stats()
    print(f"\n--- Cache Statistics ---")
    print(f"Cached judgments: {stats['cached_judgments']}")
    print(f"Note: Only 1 API call was made for 2 judgment requests!")


async def main():
    """Run all examples."""
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("LLM JUDGE EXAMPLES")
    print("="*70)
    print("Demonstrating how to use the LLM judge for answer evaluation")
    
    try:
        await example_1_answer_similarity()
        await example_2_ambiguous_question()
        await example_3_asqa_disambiguation()
        await example_4_long_form()
        await example_5_caching()
        
        print("\n" + "="*70)
        print("All examples completed!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
