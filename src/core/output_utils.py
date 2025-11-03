from pathlib import Path
from typing import Optional


def get_model_name_from_config(config_dict: dict) -> str:
    """
    Extract model name from config dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Model name (e.g., "gpt-4o-mini", "qwen2.5-3b-q4", "qwen3-4b-q4")
    """
    # Check for local LLM
    if config_dict.get("use_local_llm"):
        local_model_path = config_dict.get("local_model_path", "")
        if "Qwen3-4B" in local_model_path or "qwen3-4b" in local_model_path:
            return "qwen3-4b-q4"
        elif "qwen2.5-3b" in local_model_path:
            return "qwen2.5-3b-q4"
        else:
            return "local-unknown"

    # Check for OpenAI/API models
    model = config_dict.get("model") or config_dict.get("llm_model")
    if model:
        return model

    # Default
    return "gpt-4o-mini"


def get_organized_output_path(
    approach: str,
    retrieval_mode: str,
    model_name: str,
    is_test: bool = False,
    results_dir: str = "results"
) -> Path:
    """
    Generate organized output path following standard structure.

    Args:
        approach: RAG approach ("vanilla", "iterative", "agentic")
        retrieval_mode: Retrieval mode ("sparse", "dense", "hybrid")
        model_name: Model name (e.g., "gpt-4o-mini", "qwen2.5-3b-q4")
        is_test: Whether this is a test run
        results_dir: Base results directory (default: "results")

    Returns:
        Path object for output file

    Example:
        >>> get_organized_output_path("vanilla", "sparse", "gpt-4o-mini")
        PosixPath('results/vanilla/sparse/gpt-4o-mini/results.json')
        >>> get_organized_output_path("agentic", "hybrid", "qwen2.5-3b-q4", is_test=True)
        PosixPath('results/agentic/hybrid/qwen2.5-3b-q4/results_test.json')
    """
    # Create directory structure
    output_dir = Path(results_dir) / approach / retrieval_mode / model_name

    # Create filename
    filename = "results_test.json" if is_test else "results.json"

    return output_dir / filename


def ensure_output_directory(output_path: Path) -> None:
    """
    Ensure output directory exists.

    Args:
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
