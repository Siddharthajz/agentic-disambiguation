"""
Comparison Utility for RAG Approaches

Compares results from different RAG implementations:
- Vanilla RAG
- Iterative RAG
- Agentic Disambiguation

Generates comparative analysis and visualization.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def load_results(result_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(result_path, 'r') as f:
        return json.load(f)


def extract_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from results."""
    agg = results.get('aggregate_metrics', {})

    return {
        'mean_f1': agg.get('mean_f1', 0.0),
        'mean_d_f1': agg.get('mean_d_f1', 0.0),
        'mean_ndcg@5': agg.get('mean_ndcg@5', 0.0),
        'mean_recall@5': agg.get('mean_recall@5', 0.0),
        'mean_retrieval_time': agg.get('mean_retrieval_time', 0.0),
        'mean_generation_time': agg.get('mean_generation_time', 0.0),
        'mean_total_time': agg.get('mean_total_time', 0.0),
        'total_tokens': agg.get('total_tokens', 0),
        'num_examples': agg.get('num_examples', 0),
        'coverage_rate': agg.get('coverage_rate', 0.0)
    }


def print_comparison_table(comparison_data: Dict[str, Dict[str, float]]):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("COMPARATIVE ANALYSIS: Vanilla RAG vs Iterative RAG vs Agentic Disambiguation")
    print("="*100)

    # Metrics to compare
    metrics = [
        ('Answer Quality (F1)', 'mean_f1'),
        ('Ambiguity Handling (D-F1)', 'mean_d_f1'),
        ('Coverage Rate', 'coverage_rate'),
        ('nDCG@5', 'mean_ndcg@5'),
        ('Recall@5', 'mean_recall@5'),
        ('Retrieval Time (s)', 'mean_retrieval_time'),
        ('Generation Time (s)', 'mean_generation_time'),
        ('Total Time (s)', 'mean_total_time'),
        ('Total Tokens', 'total_tokens'),
        ('Examples', 'num_examples')
    ]

    # Print header
    approaches = list(comparison_data.keys())
    header = f"{'Metric':<30}"
    for approach in approaches:
        header += f"{approach:>20}"
    header += f"{'Best':>15}"
    print(header)
    print("-"*100)

    # Print each metric
    for metric_name, metric_key in metrics:
        row = f"{metric_name:<30}"
        values = {}

        for approach in approaches:
            value = comparison_data[approach].get(metric_key, 0.0)
            values[approach] = value

            # Format based on metric type
            if metric_key in ['total_tokens', 'num_examples']:
                row += f"{int(value):>20,}"
            elif metric_key in ['mean_retrieval_time', 'mean_generation_time', 'mean_total_time']:
                row += f"{value:>20.3f}"
            else:
                row += f"{value:>20.4f}"

        # Determine best approach (higher is better for quality metrics, lower for time/cost)
        if metric_key in ['mean_retrieval_time', 'mean_generation_time', 'mean_total_time', 'total_tokens']:
            # Lower is better
            best_approach = min(values, key=values.get) if values else "N/A"
        elif metric_key in ['num_examples']:
            best_approach = "-"
        else:
            # Higher is better
            best_approach = max(values, key=values.get) if values else "N/A"

        row += f"{best_approach:>15}"
        print(row)

    print("="*100)


def compute_improvements(comparison_data: Dict[str, Dict[str, float]], baseline: str = "vanilla"):
    """Compute percentage improvements over baseline."""
    print(f"\n" + "="*100)
    print(f"IMPROVEMENTS OVER {baseline.upper()} RAG BASELINE")
    print("="*100)

    if baseline not in comparison_data:
        print(f"Error: Baseline '{baseline}' not found in results")
        return

    baseline_metrics = comparison_data[baseline]

    # Quality metrics (higher is better)
    quality_metrics = ['mean_f1', 'mean_d_f1', 'mean_ndcg@5', 'mean_recall@5', 'coverage_rate']

    # Efficiency metrics (lower is better)
    efficiency_metrics = ['mean_total_time', 'total_tokens']

    for approach, metrics in comparison_data.items():
        if approach == baseline:
            continue

        print(f"\n{approach.upper()}:")
        print("-"*50)

        # Quality improvements
        print("Quality Improvements:")
        for metric in quality_metrics:
            baseline_val = baseline_metrics.get(metric, 0.0)
            approach_val = metrics.get(metric, 0.0)

            if baseline_val > 0:
                improvement = ((approach_val - baseline_val) / baseline_val) * 100
                direction = "↑" if improvement > 0 else "↓"
                print(f"  {metric:.<30} {improvement:>+6.2f}% {direction}")
            else:
                print(f"  {metric:.<30} N/A")

        # Efficiency comparisons
        print("\nEfficiency Comparisons:")
        for metric in efficiency_metrics:
            baseline_val = baseline_metrics.get(metric, 0.0)
            approach_val = metrics.get(metric, 0.0)

            if baseline_val > 0:
                change = ((approach_val - baseline_val) / baseline_val) * 100
                direction = "↓" if change < 0 else "↑"
                status = "better" if change < 0 else "worse"
                print(f"  {metric:.<30} {change:>+6.2f}% {direction} ({status})")
            else:
                print(f"  {metric:.<30} N/A")

    print("="*100)


def save_comparison_summary(comparison_data: Dict[str, Dict[str, float]], output_path: str):
    """Save comparison summary to JSON."""
    summary = {
        "approaches": list(comparison_data.keys()),
        "metrics": comparison_data,
        "analysis": {
            "best_answer_quality": max(comparison_data, key=lambda k: comparison_data[k].get('mean_f1', 0)),
            "best_ambiguity_handling": max(comparison_data, key=lambda k: comparison_data[k].get('mean_d_f1', 0)),
            "best_efficiency": min(comparison_data, key=lambda k: comparison_data[k].get('mean_total_time', float('inf'))),
            "lowest_cost": min(comparison_data, key=lambda k: comparison_data[k].get('total_tokens', float('inf')))
        }
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nComparison summary saved to: {output_path}")


def main():
    """Main entry point for comparison utility."""
    parser = argparse.ArgumentParser(description="Compare results from different RAG approaches")

    parser.add_argument(
        "--vanilla",
        type=str,
        default="results/vanilla_rag_results_sparse.json",
        help="Path to vanilla RAG results"
    )
    parser.add_argument(
        "--iterative",
        type=str,
        default="results/iterative_rag_results_sparse.json",
        help="Path to iterative RAG results"
    )
    parser.add_argument(
        "--agentic",
        type=str,
        default="results/agentic_disambiguation_results_hybrid.json",
        help="Path to agentic disambiguation results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison_summary.json",
        help="Path to save comparison summary"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="vanilla",
        choices=["vanilla", "iterative", "agentic"],
        help="Baseline for computing improvements"
    )

    args = parser.parse_args()

    # Load results
    comparison_data = {}

    if Path(args.vanilla).exists():
        print(f"Loading vanilla RAG results from {args.vanilla}...")
        comparison_data["vanilla"] = extract_metrics(load_results(args.vanilla))
    else:
        print(f"Warning: Vanilla RAG results not found at {args.vanilla}")

    if Path(args.iterative).exists():
        print(f"Loading iterative RAG results from {args.iterative}...")
        comparison_data["iterative"] = extract_metrics(load_results(args.iterative))
    else:
        print(f"Warning: Iterative RAG results not found at {args.iterative}")

    if Path(args.agentic).exists():
        print(f"Loading agentic disambiguation results from {args.agentic}...")
        comparison_data["agentic"] = extract_metrics(load_results(args.agentic))
    else:
        print(f"Warning: Agentic disambiguation results not found at {args.agentic}")

    if not comparison_data:
        print("Error: No results found to compare")
        return

    # Print comparison table
    print_comparison_table(comparison_data)

    # Compute improvements
    if args.baseline in comparison_data:
        compute_improvements(comparison_data, baseline=args.baseline)

    # Save summary
    save_comparison_summary(comparison_data, args.output)


if __name__ == "__main__":
    main()
