#!/usr/bin/env python3
"""
Visualize LangGraph Workflow

This script generates a Mermaid diagram of the LangGraph workflow
for agentic disambiguation.

Usage:
    python scripts/visualize_langgraph.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from core import RAGConfig
from agentic_disambiguation import LangGraphAgenticDisambiguation


def main():
    """Generate and print Mermaid diagram."""
    # Load environment
    load_dotenv()

    # Create minimal config
    config = RAGConfig(
        retrieval_mode="sparse",
        sparse_index="wikipedia-dpr",
        dense_index="wikipedia-dpr-100w.bpr-single-nq",
        dense_encoder="castorini/bpr-nq-question-encoder",
        top_k=5,
        llm_model="gpt-4o-mini",
        max_tokens=200,
        temperature=0.7,
        concurrency=10,
        use_cache=False,
        cache_dir=".cache/retrieval",
        d_f1_threshold=0.5,
        openai_api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-visualization")
    )

    # Initialize framework
    print("Initializing LangGraph framework...")
    framework = LangGraphAgenticDisambiguation(config)

    # Generate Mermaid diagram
    print("\n" + "="*80)
    print("LangGraph Workflow Visualization (Mermaid Format)")
    print("="*80 + "\n")

    try:
        # Get the compiled graph
        mermaid = framework.workflow.get_graph().draw_mermaid()
        print(mermaid)
    except Exception as e:
        print(f"Error generating Mermaid diagram: {e}")
        print("\nNote: Some LangGraph versions may not support draw_mermaid()")
        print("Fallback: Printing node structure instead\n")

        # Fallback: Print node structure
        print("```")
        print("graph TD")
        print("    START[START] --> detect_ambiguity")
        print("    detect_ambiguity{Detect Ambiguity} -->|ambiguous| generate_subqueries")
        print("    detect_ambiguity -->|not ambiguous| simple_retrieval")
        print("    generate_subqueries[Generate Sub-queries] --> generate_hyde")
        print("    generate_hyde[Generate HyDE Docs] --> retrieve_hyde")
        print("    retrieve_hyde[Retrieve with HyDE] --> synthesize_answer")
        print("    simple_retrieval[Simple Retrieval] --> synthesize_answer")
        print("    synthesize_answer[Synthesize Answer] --> END[END]")
        print("```")

    print("\n" + "="*80)
    print("Workflow Nodes:")
    print("="*80)
    print("\n1. detect_ambiguity")
    print("   - Analyzes question for ambiguity")
    print("   - Returns: is_ambiguous, confidence, reasoning")
    print("\n2. generate_subqueries (conditional)")
    print("   - Decomposes ambiguous question into 2-4 specific sub-queries")
    print("   - Returns: subqueries, reasoning")
    print("\n3. generate_hyde (conditional)")
    print("   - Creates hypothetical documents for each sub-query")
    print("   - Returns: hyde_documents (subquery → doc)")
    print("\n4. retrieve_hyde (conditional)")
    print("   - Enhanced retrieval using sub-queries + HyDE")
    print("   - Returns: retrieved_docs, retrieval_time")
    print("\n5. simple_retrieval (conditional)")
    print("   - Direct retrieval for unambiguous questions")
    print("   - Returns: retrieved_docs, retrieval_time")
    print("\n6. synthesize_answer")
    print("   - Generates comprehensive answer covering all interpretations")
    print("   - Returns: generated_answer, generation_time, total_tokens")

    print("\n" + "="*80)
    print("To visualize this diagram:")
    print("1. Copy the Mermaid code above")
    print("2. Paste into https://mermaid.live/")
    print("3. Or use a Markdown viewer with Mermaid support")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
