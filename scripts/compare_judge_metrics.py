"""
Generate comparison summary of judge metrics across all datasets.

Usage:
    python scripts/compare_judge_metrics.py
    python scripts/compare_judge_metrics.py --output judge_comparison.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from tabulate import tabulate
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_judgment_files() -> Dict[str, Dict[str, Any]]:
    """Load all judgment files from workspace root."""
    root = Path(__file__).parent.parent
    judgment_files = sorted(root.glob("*_judgments.json"))
    
    judgments = {}
    for path in judgment_files:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Skip old format files (asqa_judgments.json, etc)
                if "aggregate_metrics" in data:
                    # Extract dataset info from filename
                    name = path.stem.replace("_judgments", "")
                    judgments[name] = data
        except Exception as e:
            logger.warning(f"Failed to load {path.name}: {e}")
    
    return judgments


def build_comparison_table(judgments: Dict[str, Dict[str, Any]]) -> List[List[str]]:
    """Build comparison table of metrics."""
    headers = [
        "Dataset",
        "# Judged",
        "Sim Mean",
        "Sim Std",
        "Dis Mean",
        "Cov Rate",
        "LF Quality",
        "LF Fact",
    ]
    
    rows = []
    
    for name in sorted(judgments.keys()):
        data = judgments[name]
        metrics = data.get("aggregate_metrics", {})
        
        sim = metrics.get("similarity_scores", {})
        dis = metrics.get("disambiguation_scores", {})
        lf = metrics.get("long_form_scores", {})
        
        row = [
            name,
            str(metrics.get("total_judged", "?")),
            f"{sim.get('mean', 0):.2f}" if sim.get('count', 0) > 0 else "—",
            f"{sim.get('std', 0):.2f}" if sim.get('count', 0) > 0 else "—",
            f"{dis.get('mean', 0):.2f}" if dis.get('count', 0) > 0 else "—",
            f"{dis.get('coverage_rate', 0):.2f}" if dis.get('count', 0) > 0 else "—",
            f"{lf.get('mean_quality', 0):.2f}" if lf.get('count', 0) > 0 else "—",
            f"{lf.get('mean_factuality', 0):.2f}" if lf.get('count', 0) > 0 else "—",
        ]
        rows.append(row)
    
    return headers, rows


def analyze_by_approach(judgments: Dict[str, Dict[str, Any]]) -> Dict[str, Dict]:
    """Analyze metrics grouped by RAG approach (vanilla, agentic, iterative)."""
    approaches = {}
    
    for name, data in judgments.items():
        # Parse name: dataset_approach_retrieval_model
        parts = name.split("_")
        if len(parts) >= 2:
            # Try to identify approach
            approach = None
            if "agentic" in name:
                approach = "agentic"
            elif "iterative" in name:
                approach = "iterative"
            elif "vanilla" in name:
                approach = "vanilla"
            
            if approach:
                if approach not in approaches:
                    approaches[approach] = {
                        "datasets": [],
                        "similarity_means": [],
                        "disambiguation_means": [],
                        "long_form_means": []
                    }
                
                metrics = data.get("aggregate_metrics", {})
                approaches[approach]["datasets"].append(name)
                
                sim = metrics.get("similarity_scores", {})
                dis = metrics.get("disambiguation_scores", {})
                lf = metrics.get("long_form_scores", {})
                
                if sim.get('count', 0) > 0:
                    approaches[approach]["similarity_means"].append(sim.get('mean', 0))
                if dis.get('count', 0) > 0:
                    approaches[approach]["disambiguation_means"].append(dis.get('mean', 0))
                if lf.get('count', 0) > 0:
                    approaches[approach]["long_form_means"].append(lf.get('mean_quality', 0))
    
    # Compute aggregates
    for approach, data in approaches.items():
        if data["similarity_means"]:
            data["avg_similarity"] = sum(data["similarity_means"]) / len(data["similarity_means"])
        if data["disambiguation_means"]:
            data["avg_disambiguation"] = sum(data["disambiguation_means"]) / len(data["disambiguation_means"])
        if data["long_form_means"]:
            data["avg_long_form"] = sum(data["long_form_means"]) / len(data["long_form_means"])
    
    return approaches


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare judge metrics across datasets")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save comparison to JSON file"
    )
    args = parser.parse_args()
    
    logger.info("Loading judgment files...")
    judgments = load_judgment_files()
    
    if not judgments:
        logger.error("No judgment files found!")
        return
    
    logger.info(f"Loaded {len(judgments)} judgment files\n")
    
    # Show comparison table
    logger.info("=" * 100)
    logger.info("JUDGE METRICS COMPARISON")
    logger.info("=" * 100)
    
    headers, rows = build_comparison_table(judgments)
    table_str = tabulate(rows, headers=headers, tablefmt="grid")
    print(table_str)
    
    # Save comparison table to CSV
    csv_path = Path(__file__).parent.parent / "judge_comparison_table.csv"
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    logger.info(f"✓ Comparison table saved to {csv_path}")
    
    # Show approach analysis
    logger.info("\n" + "=" * 100)
    logger.info("ANALYSIS BY RAG APPROACH")
    logger.info("=" * 100)
    
    approaches = analyze_by_approach(judgments)
    
    approach_headers = ["Approach", "# Datasets", "Avg Similarity", "Avg Disambiguation", "Avg Long-Form"]
    approach_rows = []
    
    for approach in sorted(approaches.keys()):
        data = approaches[approach]
        row = [
            approach.upper(),
            str(len(data["datasets"])),
            f"{data.get('avg_similarity', 0):.3f}" if 'avg_similarity' in data else "—",
            f"{data.get('avg_disambiguation', 0):.3f}" if 'avg_disambiguation' in data else "—",
            f"{data.get('avg_long_form', 0):.3f}" if 'avg_long_form' in data else "—",
        ]
        approach_rows.append(row)
    
    print(tabulate(approach_rows, headers=approach_headers, tablefmt="grid"))
    
    # Key insights
    logger.info("\n" + "=" * 100)
    logger.info("KEY INSIGHTS")
    logger.info("=" * 100)
    
    if approaches:
        best_sim = max([(k, v.get('avg_similarity', 0)) for k, v in approaches.items() if 'avg_similarity' in v], key=lambda x: x[1], default=(None, 0))
        if best_sim[0]:
            logger.info(f"✓ Best semantic similarity: {best_sim[0].upper()} ({best_sim[1]:.3f})")
        
        if any('avg_disambiguation' in v for v in approaches.values()):
            best_dis = max([(k, v.get('avg_disambiguation', 0)) for k, v in approaches.items() if 'avg_disambiguation' in v], key=lambda x: x[1], default=(None, 0))
            if best_dis[0]:
                logger.info(f"✓ Best disambiguation handling: {best_dis[0].upper()} ({best_dis[1]:.3f})")
        
        if any('avg_long_form' in v for v in approaches.values()):
            best_lf = max([(k, v.get('avg_long_form', 0)) for k, v in approaches.items() if 'avg_long_form' in v], key=lambda x: x[1], default=(None, 0))
            if best_lf[0]:
                logger.info(f"✓ Best long-form quality: {best_lf[0].upper()} ({best_lf[1]:.3f})")
    
    # Save comparison if requested
    if args.output:
        output = {
            "total_datasets": len(judgments),
            "datasets": {name: {
                "metrics": data.get("aggregate_metrics", {}),
                "results_path": data.get("results_path", "")
            } for name, data in judgments.items()},
            "by_approach": approaches
        }
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"\n✓ Comparison saved to {args.output}")


if __name__ == "__main__":
    main()
