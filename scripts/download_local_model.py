#!/usr/bin/env python3
"""
Download recommended local LLM model for agentic disambiguation.

Supports multiple Qwen models in GGUF format from HuggingFace.
Works with GPU acceleration on macOS (Metal), Linux (CUDA/ROCm), and Windows (CUDA).
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Any
from urllib.request import urlretrieve


# Available model configurations
MODELS: Dict[str, Dict[str, Any]] = {
    "qwen2.5-3b-q4": {
        "name": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_mb": 2048,
        "description": "Qwen2.5-3B-Instruct (Q4_K_M quantization, ~2GB)",
        "sha256": None,
    },
    "qwen2.5-3b-q6": {
        "name": "qwen2.5-3b-instruct-q6_k.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q6_k.gguf",
        "size_mb": 2048,
        "description": "Qwen2.5-3B-Instruct (Q6_K quantization, ~2GB, higher quality)",
        "sha256": None,
    },
    "qwen3-4b-q4": {
        "name": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "size_mb": 2560,
        "description": "Qwen3-4B-Instruct-2507 (Q4_K_M quantization, ~2.5GB)",
        "sha256": None,
    },
    "qwen3-4b-q6": {
        "name": "Qwen3-4B-Instruct-2507-Q6_K.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q6_K.gguf",
        "size_mb": 3400,
        "description": "Qwen3-4B-Instruct-2507 (Q6_K quantization, ~3.3GB, higher quality)",
        "sha256": None,
    }
}

# Default recommended model
DEFAULT_MODEL = "qwen3-4b-q4"


def download_with_progress(url: str, output_path: Path):
    """Download file with progress bar."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        bar_length = 50
        filled = int(bar_length * downloaded // total_size)
        bar = '=' * filled + '-' * (bar_length - filled)

        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)

        sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
        sys.stdout.flush()

    urlretrieve(url, output_path, reporthook=report_progress)
    print()  # New line after progress bar


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verify file SHA256 checksum."""
    if not expected_sha256:
        print("No checksum provided, skipping verification")
        return True

    print("Verifying file integrity...")
    sha256 = hashlib.sha256()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)

    actual = sha256.hexdigest()

    if actual == expected_sha256:
        print("✓ Checksum verified")
        return True
    else:
        print(f"✗ Checksum mismatch!")
        print(f"  Expected: {expected_sha256}")
        print(f"  Got:      {actual}")
        return False


def list_available_models():
    """Print available model options."""
    print("\nAvailable models:")
    print("="*80)
    for model_key, model_info in MODELS.items():
        print(f"  {model_key:20} - {model_info['description']}")
    print("="*80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download local LLM model for agentic disambiguation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  qwen2.5-3b-q4    - Qwen2.5-3B-Instruct (Q4_K_M, ~2GB)
  qwen2.5-3b-q6    - Qwen2.5-3B-Instruct (Q6_K, ~2GB, higher quality)
  qwen3-4b-q4      - Qwen3-4B-Instruct-2507 (Q4_K_M, ~2.5GB) [DEFAULT]
  qwen3-4b-q6      - Qwen3-4B-Instruct-2507 (Q6_K, ~3.3GB, higher quality)
  qwen3-4b-q8      - Qwen3-4B-Instruct-2507 (Q8_0, ~4.3GB, highest quality)

Examples:
  # Download default model (Qwen3-4B-Q4)
  python scripts/download_local_model.py

  # Download specific model
  python scripts/download_local_model.py --model qwen3-4b-q6

  # Use custom URL
  python scripts/download_local_model.py --model-url <URL> --model-name <filename>
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Model to download (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for model files (default: models/)"
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default=None,
        help="Custom model URL (overrides --model)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Output filename (overrides --model)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    # List models and exit if requested
    if args.list:
        list_available_models()
        return 0

    # Determine which model config to use
    if args.model_url and args.model_name:
        # Custom model specified via URL and name
        model_config = {
            "url": args.model_url,
            "name": args.model_name,
            "size_mb": "unknown",
            "sha256": None,
        }
    else:
        # Use predefined model
        model_config = MODELS[args.model]
        # Override name or URL if specified
        if args.model_name:
            model_config = {**model_config, "name": args.model_name}
        if args.model_url:
            model_config = {**model_config, "url": args.model_url}

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / model_config["name"]

    # Check if file already exists
    if output_path.exists() and not args.force:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {output_path}")
        print(f"Size: {file_size_mb:.1f} MB")
        print("Use --force to re-download")
        return 0

    # Download model
    print("="*80)
    print("DOWNLOADING LOCAL LLM MODEL")
    print("="*80)
    print(f"Model: {model_config['name']}")
    print(f"Expected size: ~{model_config['size_mb']} MB")
    print()

    try:
        download_with_progress(model_config["url"], output_path)

        # Verify file was downloaded
        if not output_path.exists():
            print("Error: Download failed")
            return 1

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Download complete: {file_size_mb:.1f} MB")

        # Verify checksum if available
        if model_config["sha256"]:
            if not verify_checksum(output_path, model_config["sha256"]):
                print("Warning: Checksum verification failed")
                print("The downloaded file may be corrupted")
                return 1

        print()
        print("="*80)
        print("SUCCESS")
        print("="*80)
        print(f"Model saved to: {output_path}")
        print()
        print("To use this model, run:")
        print(f"  python src/agentic_disambiguation.py \\")
        print(f"    --use-local-llm \\")
        print(f"    --local-model-path {output_path} \\")
        print(f"    --single-query \"Your question here\"")

        return 0

    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
        if output_path.exists():
            output_path.unlink()
            print(f"Removed incomplete file: {output_path}")
        return 1
    except Exception as e:
        print(f"\nError downloading model: {e}")
        if output_path.exists():
            output_path.unlink()
        return 1


if __name__ == "__main__":
    sys.exit(main())
