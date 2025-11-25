import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Set

logger = logging.getLogger(__name__)


def _deterministic_hash(text: str) -> str:
    """
    Generate a deterministic hash for a string.

    Uses MD5 for consistent hashing across Python sessions.
    Python's built-in hash() uses randomization and is NOT deterministic.

    Args:
        text: String to hash

    Returns:
        Hexadecimal hash string
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


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


def load_existing_results(output_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load existing results from a file if it exists.

    Args:
        output_path: Path to results file

    Returns:
        Existing results dictionary or None if file doesn't exist
    """
    if output_path.exists():
        try:
            with open(output_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing results from {output_path}: {e}")
            return None
    return None


def get_processed_question_ids(existing_results: Dict[str, Any]) -> Set[str]:
    """
    Extract question IDs from existing results.

    Args:
        existing_results: Results dictionary with 'results' key containing list of RAGResult dicts

    Returns:
        Set of processed question IDs
    """
    if not existing_results or "results" not in existing_results:
        return set()
    
    processed_ids = set()
    for result in existing_results.get("results", []):
        if "question_id" in result:
            processed_ids.add(result["question_id"])
    
    return processed_ids


def filter_unprocessed_data(
    test_data: List[Dict[str, Any]],
    processed_ids: Set[str]
) -> List[Dict[str, Any]]:
    """
    Filter test data to only include unprocessed items.

    Args:
        test_data: List of test examples
        processed_ids: Set of already-processed question IDs

    Returns:
        Filtered list of unprocessed test examples
    """
    if not processed_ids:
        return test_data

    unprocessed = []
    for item in test_data:
        question = item.get('question', '')
        question_id = item.get('id', _deterministic_hash(question))

        if question_id not in processed_ids:
            unprocessed.append(item)

    return unprocessed


def merge_results(
    existing_results: Dict[str, Any],
    new_results: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge new results with existing results.

    Args:
        existing_results: Existing results dictionary
        new_results: List of new RAGResult dictionaries
        config: Configuration dictionary to use (from new run)

    Returns:
        Merged results dictionary
    """
    if not existing_results:
        return {
            "config": config,
            "aggregate_metrics": {},  # Will be recomputed
            "results": new_results
        }
    
    # Merge results lists
    existing_results_list = existing_results.get("results", [])
    existing_ids = {r.get("question_id") for r in existing_results_list}
    
    # Add new results, replacing any duplicates (new results take precedence)
    merged_results = existing_results_list.copy()
    for new_result in new_results:
        new_id = new_result.get("question_id")
        if new_id in existing_ids:
            # Replace existing result with new one
            merged_results = [r for r in merged_results if r.get("question_id") != new_id]
        merged_results.append(new_result)
    
    return {
        "config": config,  # Use new config
        "aggregate_metrics": {},  # Will be recomputed
        "results": merged_results
    }
