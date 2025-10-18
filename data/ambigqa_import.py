# -*- coding: utf-8 -*-
"""Local AmbigQA import script

What it does:
- Downloads AmbigNQ and NQ-open zips (if not already present)
- Extracts them to a temporary directory
- Locates train/dev/test JSON files and copies them to the target data directory
- Samples 300 records from the dev set and writes `ambignq_test.json`

Usage:
    python data/ambigqa_import.py --data-dir ./data

If the files already exist in the target data directory, the script will skip downloading.
"""

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from typing import Optional

import pandas as pd
import requests

AMBIGNQ_URL = "https://nlp.cs.washington.edu/ambigqa/data/ambignq.zip"

def download_and_unzip(url: str, target_dir: str) -> None:
    """Download a zip from `url` and extract to `target_dir`.

    Raises RuntimeError on download failure.
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading {url}...")
    resp = requests.get(url, stream=True, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download {url}: {resp.status_code}")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        print(f"Extracting to {target_dir}...")
        zf.extractall(target_dir)


def find_file(root: str, candidates) -> Optional[str]:
    """Search for the first file under `root` whose name matches any in `candidates`.

    Returns the absolute path or None if not found.
    """
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname in candidates:
                return os.path.join(dirpath, fname)
    return None


def copy_if_exists(src: str, dst: str) -> None:
    if not src or not os.path.exists(src):
        raise FileNotFoundError(f"Source file not found: {src}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(data_dir: str, sample_n: int = 300, force_download: bool = False) -> None:
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Using data directory: {data_dir}")

    # target destinations
    ambignq_train_dest = os.path.join(data_dir, "ambignq_train.json")
    ambignq_dev_dest = os.path.join(data_dir, "ambignq_dev.json")
    ambignq_test_dest = os.path.join(data_dir, "ambignq_test.json")

    need_download = force_download or not (
        os.path.exists(ambignq_train_dest)
        and os.path.exists(ambignq_dev_dest)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        if need_download:
            print("Downloading and extracting datasets...")
            download_and_unzip(AMBIGNQ_URL, tmpdir)
        else:
            print("Data files already present in target directory; skipping download.")

        # Search for expected files inside tmpdir (or if not downloaded, try data_dir)
        search_roots = [tmpdir, data_dir]

        ambignq_train_src = None
        ambignq_dev_src = None

        for root in search_roots:
            if ambignq_train_src and ambignq_dev_src:
                break
            if not ambignq_train_src:
                ambignq_train_src = find_file(root, ["train.json", "ambignq-train.json", "train.jsonl"])  # common names
            if not ambignq_dev_src:
                ambignq_dev_src = find_file(root, ["dev.json", "ambignq-dev.json", "dev.jsonl"])

        # If still not found, try to be more permissive
        if not ambignq_train_src or not ambignq_dev_src:
            # ambignq sometimes contains a single file named ambignq.json or similar
            for root in search_roots:
                if not ambignq_train_src:
                    ambignq_train_src = find_file(root, ["ambignq.json", "ambignq-train.json"])
                if not ambignq_dev_src:
                    ambignq_dev_src = find_file(root, ["ambignq.json", "ambignq-dev.json"])

        # Validate we found something
        if not ambignq_train_src or not ambignq_dev_src:
            raise FileNotFoundError(
                "Could not locate AmbigNQ train/dev files in downloaded archive or data dir."
            )

        # Copy files into data_dir
        copy_if_exists(ambignq_train_src, ambignq_train_dest)
        copy_if_exists(ambignq_dev_src, ambignq_dev_dest)

        # Create sampled test from dev
        print("Loading dev set and sampling for ambignq_test.json...")
        dev_data = load_json(ambignq_dev_dest)
        dev_df = pd.DataFrame(dev_data)

        if len(dev_df) < sample_n:
            print(f"Dev set has only {len(dev_df)} records; sampling all of them.")
            sample_n = len(dev_df)

        ambignq_test_df = dev_df.sample(n=sample_n, random_state=42)
        ambignq_test_df.to_json(ambignq_test_dest, orient="records", force_ascii=False, indent=4)
        print(f"Created sampled test set at: {ambignq_test_dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare AmbigQA datasets locally.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store data files")
    parser.add_argument("--sample-n", type=int, default=300, help="Number of samples to draw from dev for ambignq_test.json")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if files exist in data-dir")
    args = parser.parse_args()

    try:
        main(args.data_dir, sample_n=args.sample_n, force_download=args.force_download)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)