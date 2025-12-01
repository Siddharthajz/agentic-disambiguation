# -*- coding: utf-8 -*-
"""ASQA Dataset Import Script

Downloads the ASQA (Answer Summaries for Questions which are Ambiguous) dataset
from HuggingFace and converts it to the project's JSON format.

ASQA is built on top of AmbigNQ and focuses on long-form answer generation
that synthesizes information across multiple interpretations of ambiguous questions.

Dataset: https://huggingface.co/datasets/din0s/asqa
Paper: "ASQA: Factoid Questions Meet Long-Form Answers" (EMNLP 2022)

Usage:
    python data/asqa_import.py --data-dir ./data
    python data/asqa_import.py --data-dir ./data --sample-n 100

Output files:
    - asqa_train.json: Full training set (4,353 examples)
    - asqa_dev.json: Full dev set (948 examples)
    - asqa_test.json: Sampled subset for quick testing
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not installed. Run: pip install datasets")
    sys.exit(1)


def convert_asqa_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single ASQA example to project-compatible format.

    INPUT (ASQA HuggingFace format):
    {
        "ambiguous_question": str,
        "qa_pairs": [
            {
                "context": str,           # Wikipedia context passage
                "question": str,          # Disambiguated question for this interpretation
                "short_answers": [str],   # Ground truth short answers
                "wikipage": str           # Source Wikipedia page
            }
        ],
        "wikipages": [{"title": str, "url": str}],
        "annotations": [
            {
                "long_answer": str,       # Reference long-form answer
                "knowledge": [{"content": str, "wikipage": str}]
            }
        ],
        "sample_id": int
    }

    OUTPUT (Project format):
    {
        "id": str,                        # Unique identifier
        "question": str,                  # The ambiguous question (for pipeline)
        "dataset": "asqa",                # Dataset marker for auto-detection

        # === GROUND TRUTH FOR METRICS ===

        # For D-F1 (Disambiguation Coverage):
        # Each qa_pair is ONE interpretation of the ambiguous question
        "qa_pairs": [
            {
                "question": str,          # Disambiguated question
                "short_answers": [str],   # Valid answers for this interpretation
                "wikipage": str           # Source document
            }
        ],

        # For ROUGE-L (Long-form Answer Quality):
        "annotations": [
            {"long_answer": str, "knowledge": [...]}
        ],

        # For Retrieval Quality (nDCG@k, Recall@k):
        "relevant_docs": [str],           # List of relevant Wikipedia page titles

        # For F1 (pooled from all interpretations):
        "all_short_answers": [str],       # All valid short answers combined

        # Legacy compatibility fields (can be removed in future):
        "ambiguous_question": str,
        "wikipages": [...],
        "viewed_doc_titles": [str],
        "nq_answer": [str]
    }
    """
    # Extract fields from ASQA
    sample_id = example.get("sample_id", hash(example.get("ambiguous_question", "")))
    ambiguous_question = example.get("ambiguous_question", "")
    qa_pairs_raw = example.get("qa_pairs", [])
    wikipages = example.get("wikipages", [])
    annotations = example.get("annotations", [])

    # Clean up qa_pairs (remove context to reduce size, keep essential fields)
    qa_pairs = []
    all_short_answers = []
    for qa in qa_pairs_raw:
        short_answers = qa.get("short_answers", [])
        if short_answers:
            qa_pairs.append({
                "question": qa.get("question", ""),
                "short_answers": short_answers,
                "wikipage": qa.get("wikipage", "")
            })
            all_short_answers.extend(short_answers)

    # Extract relevant document titles from all sources
    relevant_docs = set()

    # From wikipages
    for wp in wikipages:
        title = wp.get("title", "").strip()
        if title:
            relevant_docs.add(title)

    # From qa_pairs
    for qa in qa_pairs:
        if qa.get("wikipage"):
            relevant_docs.add(qa["wikipage"].strip())

    # From annotations knowledge
    for ann in annotations:
        for k in ann.get("knowledge", []):
            if k.get("wikipage"):
                relevant_docs.add(k["wikipage"].strip())

    # Build the converted format
    converted = {
        # Core fields
        "id": str(sample_id),
        "question": ambiguous_question,
        "dataset": "asqa",

        # Ground truth for D-F1 (each qa_pair = one interpretation)
        "qa_pairs": qa_pairs,

        # Ground truth for ROUGE-L
        "annotations": annotations,

        # Ground truth for retrieval metrics
        "relevant_docs": sorted(relevant_docs),

        # Ground truth for F1 (all short answers pooled)
        "all_short_answers": list(set(all_short_answers)),  # Deduplicated

        # Legacy compatibility fields
        "ambiguous_question": ambiguous_question,
        "wikipages": wikipages,
        "viewed_doc_titles": sorted(relevant_docs),  # Alias for relevant_docs
        "nq_answer": qa_pairs[0]["short_answers"] if qa_pairs else [],
    }

    return converted


def download_and_convert_asqa(data_dir: str, sample_n: int = 300) -> None:
    """Download ASQA from HuggingFace and convert to project format."""

    print("Loading ASQA dataset from HuggingFace...")
    try:
        dataset = load_dataset("din0s/asqa")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative dataset source...")
        try:
            # Try alternative source if primary fails
            dataset = load_dataset("google-research-datasets/asqa")
        except Exception as e2:
            raise RuntimeError(f"Failed to load ASQA dataset: {e2}")

    print(f"Dataset loaded successfully!")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Dev: {len(dataset['dev'])} examples")

    # Convert and save train set
    train_dest = os.path.join(data_dir, "asqa_train.json")
    print(f"\nConverting train set...")
    train_converted = [convert_asqa_example(ex) for ex in dataset["train"]]
    with open(train_dest, "w", encoding="utf-8") as f:
        json.dump(train_converted, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(train_converted)} examples to {train_dest}")

    # Convert and save dev set
    dev_dest = os.path.join(data_dir, "asqa_dev.json")
    print(f"\nConverting dev set...")
    dev_converted = [convert_asqa_example(ex) for ex in dataset["dev"]]
    with open(dev_dest, "w", encoding="utf-8") as f:
        json.dump(dev_converted, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dev_converted)} examples to {dev_dest}")

    # Create sampled test set
    test_dest = os.path.join(data_dir, "asqa_test.json")
    print(f"\nCreating sampled test set ({sample_n} examples)...")
    dev_df = pd.DataFrame(dev_converted)

    if len(dev_df) < sample_n:
        print(f"Dev set has only {len(dev_df)} records; using all of them.")
        sample_n = len(dev_df)

    test_df = dev_df.sample(n=sample_n, random_state=42)
    test_df.to_json(test_dest, orient="records", force_ascii=False, indent=2)
    print(f"Saved {sample_n} examples to {test_dest}")

    # Print sample example
    print("\n" + "="*60)
    print("Sample ASQA example (converted format):")
    print("="*60)
    sample = dev_converted[0]
    print(f"ID: {sample['id']}")
    print(f"Question: {sample['question']}")
    print(f"Num interpretations (qa_pairs): {len(sample['qa_pairs'])}")
    print(f"Num annotations: {len(sample['annotations'])}")
    print(f"Ground truth docs: {sample['viewed_doc_titles'][:3]}...")
    if sample['annotations']:
        long_answer = sample['annotations'][0].get('long_answer', '')
        print(f"Reference long answer: {long_answer[:200]}...")


def main(data_dir: str, sample_n: int = 300, force_download: bool = False) -> None:
    """Main entry point for ASQA import."""
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Using data directory: {data_dir}")

    # Check if files already exist
    train_path = os.path.join(data_dir, "asqa_train.json")
    dev_path = os.path.join(data_dir, "asqa_dev.json")

    if not force_download and os.path.exists(train_path) and os.path.exists(dev_path):
        print("ASQA data files already exist. Use --force-download to re-download.")

        # Still create test set if missing or sample size changed
        test_path = os.path.join(data_dir, "asqa_test.json")
        if not os.path.exists(test_path):
            print("Creating sampled test set from existing dev data...")
            with open(dev_path, "r", encoding="utf-8") as f:
                dev_data = json.load(f)
            dev_df = pd.DataFrame(dev_data)
            actual_sample = min(sample_n, len(dev_df))
            test_df = dev_df.sample(n=actual_sample, random_state=42)
            test_df.to_json(test_path, orient="records", force_ascii=False, indent=2)
            print(f"Saved {actual_sample} examples to {test_path}")
        return

    download_and_convert_asqa(data_dir, sample_n)
    print("\nASQA import complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare ASQA dataset for RAG evaluation."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to store data files (default: ./data)"
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=300,
        help="Number of samples for asqa_test.json (default: 300)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files exist"
    )
    args = parser.parse_args()

    try:
        main(args.data_dir, sample_n=args.sample_n, force_download=args.force_download)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
