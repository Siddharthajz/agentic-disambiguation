#!/usr/bin/env python3
"""
Reorganize results directory by approach → retrieval mode → model.

Structure:
results/
├── vanilla/
│   ├── sparse/
│   │   ├── gpt-4o-mini/
│   │   ├── qwen2.5-3b-q4/
│   │   └── qwen3-4b-q4/
│   ├── dense/
│   └── hybrid/
├── iterative/
│   ├── sparse/
│   ├── dense/
│   └── hybrid/
└── agentic/
    ├── sparse/
    ├── dense/
    └── hybrid/
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional


def get_model_name(config: Dict) -> str:
    """Extract model name from config."""
    # Check for local LLM
    if config.get("use_local_llm"):
        local_model_path = config.get("local_model_path", "")
        if "Qwen3-4B" in local_model_path or "qwen3-4b" in local_model_path:
            return "qwen3-4b-q4"
        elif "qwen2.5-3b" in local_model_path:
            return "qwen2.5-3b-q4"
        else:
            # Generic local model
            return "local-unknown"

    # Check for OpenAI models
    model = config.get("model") or config.get("llm_model")
    if model:
        return model

    # Default if nothing found
    return "gpt-4o-mini"  # Assume default model if not specified


def get_approach_from_filename(filename: str) -> str:
    """Extract approach from filename."""
    filename_lower = filename.lower()

    if "agentic" in filename_lower:
        return "agentic"
    elif "iterative" in filename_lower:
        return "iterative"
    elif "vanilla" in filename_lower:
        return "vanilla"
    else:
        return "unknown"


def get_retrieval_mode(filename: str, config: Dict) -> str:
    """Extract retrieval mode from filename or config."""
    # First check config
    if "retrieval_mode" in config:
        return config["retrieval_mode"]

    # Fall back to filename parsing
    filename_lower = filename.lower()
    if "sparse" in filename_lower:
        return "sparse"
    elif "dense" in filename_lower:
        return "dense"
    elif "hybrid" in filename_lower:
        return "hybrid"
    else:
        return "unknown"


def create_new_filename(original_name: str, approach: str, mode: str, model: str) -> str:
    """Create a clean filename for the result."""
    # Check if it's a test file
    is_test = "test" in original_name.lower()

    # Base name is just the results file
    if is_test:
        return "results_test.json"
    else:
        return "results.json"


def reorganize_results(results_dir: Path, dry_run: bool = False):
    """Reorganize results directory by approach → mode → model."""

    # Find all JSON files in root of results directory
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return

    print(f"Found {len(json_files)} result files")
    print("\nAnalyzing files...\n")

    # Analyze each file
    file_mapping = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            config = data.get("config", {})
            model_name = get_model_name(config)
            approach = get_approach_from_filename(json_file.name)
            mode = get_retrieval_mode(json_file.name, config)

            # Create new path: results/approach/mode/model/
            new_dir = results_dir / approach / mode / model_name
            new_name = create_new_filename(json_file.name, approach, mode, model_name)
            new_path = new_dir / new_name

            # Store mapping
            file_mapping.append({
                "old_path": json_file,
                "new_path": new_path,
                "approach": approach,
                "mode": mode,
                "model": model_name
            })

            print(f"  {json_file.name}")
            print(f"    → Approach: {approach}")
            print(f"    → Mode: {mode}")
            print(f"    → Model: {model_name}")
            print(f"    → New path: {approach}/{mode}/{model_name}/{new_name}\n")

        except Exception as e:
            print(f"  ERROR processing {json_file.name}: {e}\n")
            continue

    # Create directories and move files
    if not dry_run:
        print("\nReorganizing files...\n")

        for mapping in file_mapping:
            # Create directory structure
            mapping["new_path"].parent.mkdir(parents=True, exist_ok=True)

            # Handle duplicate names by appending number
            new_path = mapping["new_path"]
            counter = 1
            while new_path.exists():
                stem = new_path.stem
                suffix = new_path.suffix
                new_path = new_path.parent / f"{stem}_{counter}{suffix}"
                counter += 1

            # Move file
            shutil.move(str(mapping["old_path"]), str(new_path))
            relative_path = new_path.relative_to(results_dir)
            print(f"  Moved: {mapping['old_path'].name} → {relative_path}")

        print(f"\n✓ Reorganization complete!")
        print(f"\nNew structure:")

        # Show directory tree
        for approach_dir in sorted(results_dir.glob("*/")):
            if approach_dir.name.startswith("."):
                continue
            print(f"\n  {approach_dir.name}/")
            for mode_dir in sorted(approach_dir.glob("*/")):
                print(f"    {mode_dir.name}/")
                for model_dir in sorted(mode_dir.glob("*/")):
                    files = list(model_dir.glob("*.json"))
                    print(f"      {model_dir.name}/ ({len(files)} file{'s' if len(files) != 1 else ''})")
                    for f in sorted(files):
                        print(f"        - {f.name}")
    else:
        print("\n[DRY RUN] No files were moved. Run with --execute to apply changes.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Reorganize results by approach → retrieval mode → model"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory to reorganize (default: results)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files (default is dry-run)"
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: {args.results_dir} does not exist")
        return 1

    reorganize_results(args.results_dir, dry_run=not args.execute)
    return 0


if __name__ == "__main__":
    exit(main())
