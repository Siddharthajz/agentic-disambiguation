#!/usr/bin/env python3
"""
Quick test of the disambiguation fix with a Star Wars example.
This tests that the new prompt correctly identifies that answer "1977"
does not cover interpretations asking for 1999, 2002, etc.
"""

import asyncio
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.judge import OpenAIJudge

async def test_star_wars_disambiguation():
    """Test the fixed disambiguation judgment on Star Wars example."""
    
    judge = OpenAIJudge(model="gpt-4o-mini")
    
    question = "When did Star Wars come out?"
    generated_answer = "1977"
    
    # Simulated qa_pairs for Star Wars (simplified from actual data)
    qa_pairs = [
        {"question": "When did Star Wars: A New Hope come out?", "short_answers": ["1977", "May 25, 1977"]},
        {"question": "When did Star Wars: The Empire Strikes Back come out?", "short_answers": ["1980", "May 21, 1980"]},
        {"question": "When did Star Wars: Return of the Jedi come out?", "short_answers": ["1983", "May 25, 1983"]},
        {"question": "When did Star Wars: The Phantom Menace come out?", "short_answers": ["1999", "May 19, 1999"]},
        {"question": "When did Star Wars: Attack of the Clones come out?", "short_answers": ["2002", "May 16, 2002"]},
        {"question": "When did Star Wars: Revenge of the Sith come out?", "short_answers": ["2005", "May 19, 2005"]},
    ]
    
    print("=" * 80)
    print("Testing Disambiguation Fix")
    print("=" * 80)
    print(f"\nQuestion: {question}")
    print(f"Generated Answer: {generated_answer}")
    print(f"Number of interpretations: {len(qa_pairs)}")
    print("\nInterpretations:")
    for i, qa in enumerate(qa_pairs, 1):
        print(f"  {i}. {qa['question']}")
        print(f"     Valid answers: {', '.join(qa['short_answers'])}")
    
    print("\n" + "=" * 80)
    print("Running disambiguation judgment...")
    print("=" * 80)
    
    result = await judge.judge_disambiguation(
        question=question,
        generated_answer=generated_answer,
        ground_truth_interpretations=qa_pairs,
        dataset="ambignq"
    )
    
    print(f"\n✓ Judgment completed in {result['generation_time']:.2f}s ({result['total_tokens']} tokens)")
    print(f"\nResults:")
    print(f"  Total interpretations: {result['total_interpretations']}")
    print(f"  Interpretations covered: {result['interpretations_covered']}")
    print(f"  Disambiguation score: {result['disambiguation_score']:.2f}")
    print(f"  Covered indices: {result['covered_interpretations']}")
    print(f"  Missing indices: {result['missing_interpretations']}")
    
    print(f"\nPer-interpretation analysis:")
    for analysis in result.get('per_interpretation_analysis', []):
        idx = analysis['index']
        status = "✓ COVERED" if analysis['is_covered'] else "✗ NOT COVERED"
        print(f"\n  [{idx}] {status}")
        print(f"      Required: {analysis.get('required_answer', 'N/A')}")
        print(f"      Generated says: {analysis.get('generated_content', 'N/A')}")
        print(f"      Reason: {analysis.get('reasoning', 'N/A')}")
    
    print(f"\nOverall: {result['reasoning']}")
    
    # Validation checks
    print("\n" + "=" * 80)
    print("Validation Checks")
    print("=" * 80)
    
    checks = [
        ("Covered only 1 interpretation (1977 release)", result['interpretations_covered'] == 1),
        ("Interpretation 0 is covered", 0 in result['covered_interpretations']),
        ("Interpretations 1-5 are NOT covered", all(i in result['missing_interpretations'] for i in range(1, 6))),
        ("Disambiguation score is ~0.17 (1/6)", abs(result['disambiguation_score'] - (1/6)) < 0.05),
    ]
    
    all_passed = True
    for check_name, check_result in checks:
        status = "✓ PASS" if check_result else "✗ FAIL"
        print(f"{status}: {check_name}")
        if not check_result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All validation checks passed! The fix is working correctly.")
    else:
        print("✗ Some checks failed. Review the results above.")
    print("=" * 80)
    
    return result

if __name__ == "__main__":
    result = asyncio.run(test_star_wars_disambiguation())
    
    # Save full result for inspection
    with open("/tmp/test_disambiguation_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull result saved to /tmp/test_disambiguation_result.json")
