import math
import re
import string
from collections import Counter
from typing import List, Dict, Any, Set, Tuple, Optional
from enum import Enum
import numpy as np

# Optional import for ROUGE-L (ASQA evaluation)
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

# Optional import for transformers (ASQA QA-based evaluation)
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class DatasetType(Enum):
    """Supported dataset types for evaluation."""
    AMBIGNQ = "ambignq"
    ASQA = "asqa"


# =============================================================================
# Ground Truth Structure Documentation
# =============================================================================
"""
GROUND TRUTH STRUCTURE FOR EACH METRIC:

1. F1 SCORE (Answer Quality)
   - Measures: Token overlap between prediction and ANY valid answer
   - AmbigNQ Ground Truth: Pool of ALL short answers from all interpretations
     - From: annotations[].answer (singleAnswer) OR annotations[].qaPairs[].answer (multipleQAs)
     - Plus: nq_answer (original Natural Questions answer)
   - ASQA Ground Truth: Pool of ALL short answers from qa_pairs
     - From: qa_pairs[].short_answers
     - Plus: nq_answer (first short answer, for compatibility)
   - Scoring: max(F1(prediction, gt) for gt in all_answers)

2. D-F1 SCORE (Disambiguation Coverage)
   - Measures: What fraction of interpretations are "covered" by the prediction
   - Each interpretation has a SET of valid answers (any one is acceptable)
   - An interpretation is "covered" if F1(prediction, any_answer_in_set) >= threshold
   - AmbigNQ Ground Truth: List of answer sets, one per interpretation
     - singleAnswer: One interpretation with answer list
     - multipleQAs: Multiple interpretations, each qaPair is one interpretation
   - ASQA Ground Truth: List of answer sets from qa_pairs
     - Each qa_pair represents one interpretation
     - QA-based evaluation: Use RoBERTa to extract answer from prediction, compare to short_answers
   - Scoring: covered_interpretations / total_interpretations

3. ROUGE-L (Long-form Answer Quality) - ASQA ONLY
   - Measures: Longest common subsequence overlap with reference long answers
   - ASQA Ground Truth: annotations[].long_answer
   - Scoring: max(ROUGE-L(prediction, ref) for ref in long_answers)

4. DR-F1 (ASQA Primary Metric) - ASQA ONLY
   - Measures: Combined disambiguation and answer quality
   - Formula: sqrt(D-F1 * ROUGE-L) (geometric mean)

5. nDCG@k, Recall@k (Retrieval Quality)
   - Measures: Quality of retrieved documents
   - AmbigNQ Ground Truth: viewed_doc_titles, nq_doc_title, used_queries[].results[].title
   - ASQA Ground Truth: wikipages[].title, annotations[].knowledge[].wikipage, qa_pairs[].wikipage
"""


# =============================================================================
# Ground Truth Extraction - Explicit and Documented
# =============================================================================

class GroundTruth:
    """
    Standardized ground truth container for evaluation.

    This class provides explicit access to all ground truth data needed
    for evaluation metrics, with clear documentation of sources.
    """

    def __init__(
        self,
        dataset_type: DatasetType,
        # For F1: All acceptable short answers (pooled)
        all_short_answers: List[str],
        # For D-F1: List of answer sets (one set per interpretation)
        interpretation_answer_sets: List[List[str]],
        # For D-F1 (ASQA QA-based): qa_pairs with questions and answers
        qa_pairs: Optional[List[Dict[str, Any]]] = None,
        # For ROUGE-L: Long-form reference answers (ASQA only)
        long_answer_references: Optional[List[str]] = None,
        # For Retrieval: Relevant document titles
        relevant_doc_titles: Optional[Set[str]] = None,
    ):
        self.dataset_type = dataset_type
        self.all_short_answers = all_short_answers
        self.interpretation_answer_sets = interpretation_answer_sets
        self.qa_pairs = qa_pairs or []
        self.long_answer_references = long_answer_references or []
        self.relevant_doc_titles = relevant_doc_titles or set()

    @property
    def num_interpretations(self) -> int:
        """Number of distinct interpretations of the ambiguous question."""
        return len(self.interpretation_answer_sets)

    def __repr__(self) -> str:
        return (
            f"GroundTruth(dataset={self.dataset_type.value}, "
            f"interpretations={self.num_interpretations}, "
            f"short_answers={len(self.all_short_answers)}, "
            f"long_answers={len(self.long_answer_references)}, "
            f"relevant_docs={len(self.relevant_doc_titles)})"
        )


def extract_ground_truth(reference_item: Dict[str, Any]) -> GroundTruth:
    """
    Extract all ground truth data from a reference item.

    This is the SINGLE entry point for ground truth extraction.
    Auto-detects dataset type and extracts appropriate fields.

    Args:
        reference_item: Raw data item from AmbigNQ or ASQA dataset

    Returns:
        GroundTruth object with all evaluation data
    """
    dataset_type = detect_dataset_type(reference_item)

    if dataset_type == DatasetType.ASQA:
        return _extract_ground_truth_asqa(reference_item)
    else:
        return _extract_ground_truth_ambignq(reference_item)


def _extract_ground_truth_ambignq(item: Dict[str, Any]) -> GroundTruth:
    """Extract ground truth from AmbigNQ format."""

    # 1. Extract interpretation answer sets from annotations
    interpretation_sets = []
    annotations = item.get('annotations', [])

    for annotation in annotations:
        ann_type = annotation.get('type', '')

        if ann_type == 'singleAnswer':
            # Single interpretation with possibly multiple valid answers
            answers = annotation.get('answer', [])
            if isinstance(answers, str):
                answers = [answers]
            if answers:
                interpretation_sets.append(answers)

        elif ann_type == 'multipleQAs':
            # Multiple interpretations, each qaPair is one
            qa_pairs = annotation.get('qaPairs', [])
            for qa in qa_pairs:
                answers = qa.get('answer', [])
                if isinstance(answers, str):
                    answers = [answers]
                if answers:
                    interpretation_sets.append(answers)

    # 2. Pool all short answers for F1
    all_short_answers = []
    for answer_set in interpretation_sets:
        all_short_answers.extend(answer_set)

    # Add nq_answer (original NQ answer)
    nq_answer = item.get('nq_answer', [])
    if isinstance(nq_answer, str):
        all_short_answers.append(nq_answer)
    elif isinstance(nq_answer, list):
        all_short_answers.extend(nq_answer)

    # 3. Extract relevant doc titles for retrieval
    relevant_titles = set()

    # From viewed_doc_titles
    viewed = item.get('viewed_doc_titles', [])
    if isinstance(viewed, list):
        relevant_titles.update(t.strip() for t in viewed if t)

    # From nq_doc_title
    nq_title = item.get('nq_doc_title', '').strip()
    if nq_title:
        relevant_titles.add(nq_title)

    # From used_queries results
    for query_data in item.get('used_queries', []):
        for result in query_data.get('results', []):
            title = result.get('title', '').strip()
            if title:
                relevant_titles.add(title)

    return GroundTruth(
        dataset_type=DatasetType.AMBIGNQ,
        all_short_answers=all_short_answers,
        interpretation_answer_sets=interpretation_sets,
        qa_pairs=None,  # AmbigNQ doesn't use QA-based D-F1
        long_answer_references=None,  # AmbigNQ is short-answer only
        relevant_doc_titles=relevant_titles,
    )


def _extract_ground_truth_asqa(item: Dict[str, Any]) -> GroundTruth:
    """
    Extract ground truth from ASQA format.

    Supports both:
    - New format (with all_short_answers, relevant_docs pre-computed)
    - Legacy format (requires extraction from nested fields)
    """
    qa_pairs = item.get('qa_pairs', [])
    annotations = item.get('annotations', [])

    # 1. Extract interpretation answer sets from qa_pairs
    # Each qa_pair represents ONE interpretation of the ambiguous question
    interpretation_sets = []
    for qa in qa_pairs:
        short_answers = qa.get('short_answers', [])
        if isinstance(short_answers, str):
            short_answers = [short_answers]
        if short_answers:
            interpretation_sets.append(short_answers)

    # 2. Get all short answers for F1
    # Prefer pre-computed field if available (new format)
    if 'all_short_answers' in item and item['all_short_answers']:
        all_short_answers = item['all_short_answers']
    else:
        # Extract from interpretation sets (legacy format)
        all_short_answers = []
        for answer_set in interpretation_sets:
            all_short_answers.extend(answer_set)

        # Add nq_answer for compatibility
        nq_answer = item.get('nq_answer', [])
        if isinstance(nq_answer, str):
            all_short_answers.append(nq_answer)
        elif isinstance(nq_answer, list):
            all_short_answers.extend(nq_answer)

    # 3. Extract long-form reference answers (from annotations)
    long_answers = []
    for ann in annotations:
        long_answer = ann.get('long_answer', '')
        if long_answer and isinstance(long_answer, str):
            long_answers.append(long_answer)

    # 4. Get relevant document titles
    # Prefer pre-computed field if available (new format)
    if 'relevant_docs' in item and item['relevant_docs']:
        relevant_titles = set(item['relevant_docs'])
    else:
        # Extract from multiple sources (legacy format)
        relevant_titles = set()

        # From wikipages
        for wp in item.get('wikipages', []):
            title = wp.get('title', '').strip()
            if title:
                relevant_titles.add(title)

        # From annotations knowledge
        for ann in annotations:
            for k in ann.get('knowledge', []):
                wikipage = k.get('wikipage', '').strip()
                if wikipage:
                    relevant_titles.add(wikipage)

        # From qa_pairs wikipage
        for qa in qa_pairs:
            wikipage = qa.get('wikipage')
            if wikipage and isinstance(wikipage, str):
                relevant_titles.add(wikipage.strip())

        # Fallback: viewed_doc_titles
        viewed = item.get('viewed_doc_titles', [])
        if isinstance(viewed, list):
            relevant_titles.update(t.strip() for t in viewed if t)

    return GroundTruth(
        dataset_type=DatasetType.ASQA,
        all_short_answers=all_short_answers,
        interpretation_answer_sets=interpretation_sets,
        qa_pairs=qa_pairs,  # For QA-based D-F1 evaluation
        long_answer_references=long_answers,
        relevant_doc_titles=relevant_titles,
    )


def detect_dataset_type(reference_item: Dict[str, Any]) -> DatasetType:
    """
    Auto-detect dataset type from reference item structure.

    Args:
        reference_item: Reference data item

    Returns:
        DatasetType enum value
    """
    # Check explicit dataset marker
    if reference_item.get("dataset") == "asqa":
        return DatasetType.ASQA

    # Check for ASQA-specific fields
    if "ambiguous_question" in reference_item and "qa_pairs" in reference_item:
        return DatasetType.ASQA

    # Check for ASQA-style annotations (long_answer field)
    annotations = reference_item.get("annotations", [])
    if annotations and isinstance(annotations[0], dict):
        if "long_answer" in annotations[0]:
            return DatasetType.ASQA

    return DatasetType.AMBIGNQ


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.

    Args:
        text: Raw answer text

    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def extract_tokens(text: str) -> List[str]:
    """Extract tokens from normalized text."""
    return normalize_answer(text).split()


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score between prediction and ground truth.

    Args:
        prediction: Predicted answer
        ground_truth: Reference answer

    Returns:
        F1 score (0.0 to 1.0)
    """
    pred_tokens = extract_tokens(prediction)
    gt_tokens = extract_tokens(ground_truth)

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_max_f1(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute max F1 score against multiple ground truth answers.

    Args:
        prediction: Predicted answer
        ground_truths: List of reference answers

    Returns:
        Maximum F1 score
    """
    if not ground_truths:
        return 0.0

    return max(compute_f1(prediction, gt) for gt in ground_truths)


def extract_answers_from_annotations(annotations: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Extract all answer sets from AmbigNQ annotations.

    Args:
        annotations: List of annotation dictionaries

    Returns:
        List of answer sets (each set corresponds to one interpretation)
    """
    answer_sets = []

    for annotation in annotations:
        if annotation.get('type') == 'singleAnswer':
            # Single interpretation
            answers = annotation.get('answer', [])
            if isinstance(answers, str):
                answers = [answers]
            answer_sets.append(answers)
        elif annotation.get('type') == 'multipleQAs':
            # Multiple interpretations
            qas = annotation.get('qaPairs', [])
            for qa in qas:
                answers = qa.get('answer', [])
                if isinstance(answers, str):
                    answers = [answers]
                answer_sets.append(answers)

    return answer_sets


def compute_disambiguation_f1(
    prediction: str,
    annotations: List[Dict[str, Any]],
    threshold: float = 0.5
) -> Tuple[float, int, int]:
    """
    Compute Disambiguation F1 (D-F1) score.

    D-F1 measures how well the system covers all plausible interpretations
    of an ambiguous question.

    Args:
        prediction: Generated answer (may contain multiple answers)
        annotations: AmbigNQ annotations with multiple interpretations
        threshold: F1 threshold to consider an interpretation "covered"

    Returns:
        Tuple of (D-F1 score, num_covered, num_total_interpretations)
    """
    answer_sets = extract_answers_from_annotations(annotations)

    if not answer_sets:
        return 0.0, 0, 0

    # Check which interpretations are covered
    covered = 0
    for answer_set in answer_sets:
        # An interpretation is "covered" if prediction has F1 >= threshold with any answer in the set
        max_f1 = compute_max_f1(prediction, answer_set)
        if max_f1 >= threshold:
            covered += 1

    total_interpretations = len(answer_sets)

    # D-F1 is the coverage ratio
    d_f1 = covered / total_interpretations if total_interpretations > 0 else 0.0

    return d_f1, covered, total_interpretations


def compute_ndcg_at_k(
    retrieved_docs: List[Dict[str, Any]],
    relevant_doc_titles: Set[str],
    k: int = 5
) -> float:
    """
    Compute normalized Discounted Cumulative Gain at k (nDCG@k).

    Args:
        retrieved_docs: List of retrieved documents (ordered by rank)
        relevant_doc_titles: Set of relevant document titles
        k: Cutoff position

    Returns:
        nDCG@k score (0.0 to 1.0)
    """
    if not relevant_doc_titles:
        return 0.0

    # Compute DCG@k
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k]):
        rank = i + 1
        doc_title = doc.get('title', '').strip()

        # Relevance: 1 if document is relevant, 0 otherwise
        relevance = 1.0 if doc_title in relevant_doc_titles else 0.0

        # DCG formula: sum(rel_i / log2(i + 1))
        dcg += relevance / np.log2(rank + 1)

    # Compute ideal DCG (all relevant docs at top positions)
    num_relevant = len(relevant_doc_titles)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, num_relevant)))

    if idcg == 0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


def compute_recall_at_k(
    retrieved_docs: List[Dict[str, Any]],
    relevant_doc_titles: Set[str],
    k: int = 5
) -> float:
    """
    Compute Recall@k.

    Args:
        retrieved_docs: List of retrieved documents (ordered by rank)
        relevant_doc_titles: Set of relevant document titles
        k: Cutoff position

    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not relevant_doc_titles:
        return 0.0

    # Get titles of top-k retrieved docs
    retrieved_titles = {
        doc.get('title', '').strip()
        for doc in retrieved_docs[:k]
    }

    # Count how many relevant docs are in top-k
    num_relevant_retrieved = len(retrieved_titles & relevant_doc_titles)

    # Recall = relevant retrieved / total relevant
    recall = num_relevant_retrieved / len(relevant_doc_titles)

    return recall


def extract_relevant_doc_titles(item: Dict[str, Any]) -> Set[str]:
    """
    Extract relevant document titles from AmbigNQ item.

    Args:
        item: AmbigNQ test item

    Returns:
        Set of relevant document titles
    """
    relevant_titles = set()

    # Add viewed doc titles
    viewed = item.get('viewed_doc_titles', [])
    if isinstance(viewed, list):
        relevant_titles.update(title.strip() for title in viewed)

    # Add doc titles from used queries
    queries = item.get('used_queries', [])
    for query_data in queries:
        results = query_data.get('results', [])
        for result in results:
            title = result.get('title', '').strip()
            if title:
                relevant_titles.add(title)

    # Add NQ doc title
    nq_title = item.get('nq_doc_title', '').strip()
    if nq_title:
        relevant_titles.add(nq_title)

    return relevant_titles


# =============================================================================
# ASQA-Specific Functions
# =============================================================================

def extract_answers_from_asqa(reference_item: Dict[str, Any]) -> List[List[str]]:
    """
    Extract answer sets from ASQA qa_pairs for D-F1 computation.

    Each qa_pair represents one interpretation of the ambiguous question.

    Args:
        reference_item: ASQA data item

    Returns:
        List of answer sets (each set corresponds to one interpretation)
    """
    answer_sets = []

    # Primary source: qa_pairs field
    qa_pairs = reference_item.get("qa_pairs", [])
    for qa in qa_pairs:
        short_answers = qa.get("short_answers", [])
        if short_answers:
            if isinstance(short_answers, str):
                short_answers = [short_answers]
            answer_sets.append(short_answers)

    # Fallback: ambignq_annotations (from our import conversion)
    if not answer_sets:
        ambignq_annotations = reference_item.get("ambignq_annotations", [])
        answer_sets = extract_answers_from_annotations(ambignq_annotations)

    return answer_sets


def extract_long_answer_references(reference_item: Dict[str, Any]) -> List[str]:
    """
    Extract long-form reference answers from ASQA annotations.

    Args:
        reference_item: ASQA data item

    Returns:
        List of reference long-form answers
    """
    long_answers = []

    annotations = reference_item.get("annotations", [])
    for ann in annotations:
        long_answer = ann.get("long_answer", "")
        if long_answer and isinstance(long_answer, str):
            long_answers.append(long_answer)

    return long_answers


def extract_relevant_docs_asqa(reference_item: Dict[str, Any]) -> Set[str]:
    """
    Extract relevant document titles from ASQA item.

    Args:
        reference_item: ASQA data item

    Returns:
        Set of relevant document titles
    """
    relevant_titles = set()

    # From wikipages field
    wikipages = reference_item.get("wikipages", [])
    for wp in wikipages:
        title = wp.get("title", "").strip()
        if title:
            relevant_titles.add(title)

    # From annotations knowledge field
    annotations = reference_item.get("annotations", [])
    for ann in annotations:
        knowledge = ann.get("knowledge", [])
        for k in knowledge:
            wikipage = k.get("wikipage", "").strip()
            if wikipage:
                relevant_titles.add(wikipage)

    # From qa_pairs context
    qa_pairs = reference_item.get("qa_pairs", [])
    for qa in qa_pairs:
        wikipage = qa.get("wikipage")
        if wikipage and isinstance(wikipage, str):
            relevant_titles.add(wikipage.strip())

    # Fallback: viewed_doc_titles (from our import conversion)
    viewed = reference_item.get("viewed_doc_titles", [])
    if isinstance(viewed, list):
        relevant_titles.update(title.strip() for title in viewed if title)

    return relevant_titles


def compute_rouge_l(prediction: str, references: List[str]) -> float:
    """
    Compute ROUGE-L score between prediction and reference answers.

    ROUGE-L measures longest common subsequence overlap, suitable for
    evaluating long-form answers.

    Args:
        prediction: Generated long-form answer
        references: List of reference long-form answers

    Returns:
        Maximum ROUGE-L F1 score (0.0 to 1.0)
    """
    if not ROUGE_AVAILABLE:
        # Fallback to simple token overlap if rouge_score not installed
        return _compute_rouge_l_fallback(prediction, references)

    if not references:
        return 0.0

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    max_score = 0.0
    for ref in references:
        if not ref:
            continue
        scores = scorer.score(ref, prediction)
        rouge_l_f1 = scores['rougeL'].fmeasure
        max_score = max(max_score, rouge_l_f1)

    return max_score


def _compute_rouge_l_fallback(prediction: str, references: List[str]) -> float:
    """
    Fallback ROUGE-L computation using LCS without external library.

    Args:
        prediction: Generated answer
        references: List of reference answers

    Returns:
        Maximum ROUGE-L F1 score
    """
    if not references:
        return 0.0

    def lcs_length(s1: List[str], s2: List[str]) -> int:
        """Compute LCS length between two token lists."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    max_score = 0.0
    for ref in references:
        if not ref:
            continue
        ref_tokens = normalize_answer(ref).split()
        if not ref_tokens:
            continue

        lcs_len = lcs_length(pred_tokens, ref_tokens)

        precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0

        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
            max_score = max(max_score, f1)

    return max_score


def compute_dr_f1(d_f1: float, rouge_l: float) -> float:
    """
    Compute DR-F1 (Disambiguation-ROUGE F1) score.

    DR-F1 is the primary ASQA metric, combining disambiguation coverage
    with long-form answer quality using the geometric mean.

    Formula: DR-F1 = sqrt(D-F1 × ROUGE-L)

    Args:
        d_f1: Disambiguation F1 score
        rouge_l: ROUGE-L score

    Returns:
        DR-F1 score (0.0 to 1.0)
    """
    return math.sqrt(d_f1 * rouge_l)


class ASQAQAEvaluator:
    """
    QA-based evaluator for ASQA disambiguation metric.

    Uses a RoBERTa-based QA model to extract answers from the generated
    long-form answer, then checks if the extracted answer matches the
    ground truth short answer.

    This is the standard ASQA evaluation method that avoids false positives
    from naive string matching.

    The model is cached locally in 'models/roberta-squad2' to avoid
    re-downloading on each run.
    """

    _instance = None
    _qa_pipeline = None

    # Model configuration
    MODEL_NAME = "deepset/roberta-base-squad2"
    LOCAL_MODEL_DIR = "models/roberta-squad2"

    def __new__(cls):
        """Singleton pattern to ensure model is loaded only once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the QA pipeline (lazy loading)."""
        pass

    def _get_model_path(self) -> str:
        """
        Get the model path, preferring local cache.

        Returns:
            Path to local model if exists, otherwise HuggingFace model name
        """
        import os
        from pathlib import Path

        # Check if local model exists
        local_path = Path(self.LOCAL_MODEL_DIR)
        if local_path.exists() and (local_path / "config.json").exists():
            return str(local_path)

        return self.MODEL_NAME

    def _save_model_locally(self):
        """Save the loaded model to local cache for future use."""
        import os
        from pathlib import Path

        local_path = Path(self.LOCAL_MODEL_DIR)

        # Only save if not already local
        if not local_path.exists() or not (local_path / "config.json").exists():
            try:
                local_path.mkdir(parents=True, exist_ok=True)
                # Save model and tokenizer
                self._qa_pipeline.model.save_pretrained(str(local_path))
                self._qa_pipeline.tokenizer.save_pretrained(str(local_path))
                print(f"Saved QA model to {local_path} for future use")
            except Exception as e:
                print(f"Warning: Could not save model locally: {e}")

    def _load_model(self):
        """
        Load the QA model lazily on first use.

        Loads from local cache if available, otherwise downloads from HuggingFace
        and saves locally for future runs.
        """
        if self._qa_pipeline is None:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError(
                    "transformers package not installed. "
                    "Install with: pip install transformers"
                )

            model_path = self._get_model_path()
            is_local = model_path != self.MODEL_NAME

            if is_local:
                print(f"Loading QA model from local cache: {model_path}")
            else:
                print(f"Downloading QA model: {self.MODEL_NAME} (will be cached locally)")

            self._qa_pipeline = hf_pipeline(
                "question-answering",
                model=model_path,
                tokenizer=model_path,
                device=-1  # CPU, change to 0 for GPU
            )

            # Save locally if downloaded from HuggingFace
            if not is_local:
                self._save_model_locally()

    def extract_answer(self, context: str, question: str) -> Tuple[str, float]:
        """
        Extract an answer from context given a question.

        Args:
            context: The generated long-form answer (used as context)
            question: The disambiguated question from qa_pairs

        Returns:
            Tuple of (extracted_answer, confidence_score)
        """
        self._load_model()

        try:
            result = self._qa_pipeline(question=question, context=context)
            return result['answer'], result['score']
        except Exception:
            return "", 0.0

    def check_answer_match(
        self,
        extracted_answer: str,
        ground_truth_answers: List[str],
        threshold: float = 0.5
    ) -> bool:
        """
        Check if extracted answer matches any ground truth answer.

        Uses F1 overlap to allow for partial matches.

        Args:
            extracted_answer: Answer extracted by QA model
            ground_truth_answers: List of acceptable answers
            threshold: F1 threshold to consider a match

        Returns:
            True if extracted answer matches ground truth
        """
        if not extracted_answer or not ground_truth_answers:
            return False

        max_f1 = compute_max_f1(extracted_answer, ground_truth_answers)
        return max_f1 >= threshold

    def evaluate_disambiguation(
        self,
        generated_answer: str,
        qa_pairs: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> Tuple[float, int, int]:
        """
        Evaluate disambiguation coverage using QA extraction.

        For each qa_pair, extract an answer from the generated text
        and check if it matches the ground truth.

        Args:
            generated_answer: The generated long-form answer
            qa_pairs: List of qa_pairs with questions and short_answers
            threshold: F1 threshold for answer matching

        Returns:
            Tuple of (D-F1 score, num_covered, num_total)
        """
        if not qa_pairs:
            return 0.0, 0, 0

        covered = 0
        for qa in qa_pairs:
            question = qa.get("question", "")
            short_answers = qa.get("short_answers", [])

            if isinstance(short_answers, str):
                short_answers = [short_answers]

            if not question or not short_answers:
                continue

            # Extract answer using QA model
            extracted, confidence = self.extract_answer(generated_answer, question)

            # Check if extracted answer matches ground truth
            if self.check_answer_match(extracted, short_answers, threshold):
                covered += 1

        total = len(qa_pairs)
        d_f1 = covered / total if total > 0 else 0.0

        return d_f1, covered, total


# Global singleton instance
_asqa_qa_evaluator: Optional[ASQAQAEvaluator] = None


def get_asqa_qa_evaluator() -> ASQAQAEvaluator:
    """Get the singleton ASQA QA evaluator instance."""
    global _asqa_qa_evaluator
    if _asqa_qa_evaluator is None:
        _asqa_qa_evaluator = ASQAQAEvaluator()
    return _asqa_qa_evaluator


def compute_disambiguation_f1_asqa(
    prediction: str,
    reference_item: Dict[str, Any],
    threshold: float = 0.5,
    use_qa_model: bool = True
) -> Tuple[float, int, int]:
    """
    Compute D-F1 for ASQA items using qa_pairs.

    Uses a QA model to extract answers from the generated text and
    check if they match the ground truth short answers. This avoids
    false positives from naive string matching.

    Args:
        prediction: Generated answer (long-form)
        reference_item: ASQA data item
        threshold: F1 threshold to consider interpretation covered
        use_qa_model: If True, use RoBERTa QA model for extraction.
                     If False, fall back to naive token overlap.

    Returns:
        Tuple of (D-F1 score, num_covered, num_total_interpretations)
    """
    qa_pairs = reference_item.get("qa_pairs", [])

    # Use QA-based evaluation if available and requested
    if use_qa_model and TRANSFORMERS_AVAILABLE and qa_pairs:
        evaluator = get_asqa_qa_evaluator()
        return evaluator.evaluate_disambiguation(prediction, qa_pairs, threshold)

    # Fallback to naive token overlap
    answer_sets = extract_answers_from_asqa(reference_item)

    if not answer_sets:
        return 0.0, 0, 0

    covered = 0
    for answer_set in answer_sets:
        max_f1 = compute_max_f1(prediction, answer_set)
        if max_f1 >= threshold:
            covered += 1

    total_interpretations = len(answer_sets)
    d_f1 = covered / total_interpretations if total_interpretations > 0 else 0.0

    return d_f1, covered, total_interpretations


class RAGEvaluator:
    """Comprehensive evaluator for RAG systems on AmbigNQ and ASQA."""

    def __init__(self, k: int = 5, d_f1_threshold: float = 0.5):
        """
        Initialize evaluator.

        Args:
            k: Cutoff for IR metrics (nDCG@k, Recall@k)
            d_f1_threshold: F1 threshold for D-F1 computation
        """
        self.k = k
        self.d_f1_threshold = d_f1_threshold

    def evaluate_single(
        self,
        prediction: str,
        retrieved_docs: List[Dict[str, Any]],
        reference_item: Dict[str, Any],
        retrieval_time: float = 0.0,
        generation_time: float = 0.0,
        total_tokens: int = 0
    ) -> Dict[str, Any]:
        """
        Evaluate a single prediction. Auto-detects dataset type.

        Args:
            prediction: Generated answer
            retrieved_docs: Retrieved documents
            reference_item: Reference data from AmbigNQ or ASQA
            retrieval_time: Time spent on retrieval (seconds)
            generation_time: Time spent on generation (seconds)
            total_tokens: Total tokens processed

        Returns:
            Dictionary of metrics
        """
        dataset_type = detect_dataset_type(reference_item)

        if dataset_type == DatasetType.ASQA:
            return self._evaluate_asqa(
                prediction, retrieved_docs, reference_item,
                retrieval_time, generation_time, total_tokens
            )
        else:
            return self._evaluate_ambignq(
                prediction, retrieved_docs, reference_item,
                retrieval_time, generation_time, total_tokens
            )

    def _evaluate_ambignq(
        self,
        prediction: str,
        retrieved_docs: List[Dict[str, Any]],
        reference_item: Dict[str, Any],
        retrieval_time: float,
        generation_time: float,
        total_tokens: int
    ) -> Dict[str, Any]:
        """
        Evaluate for AmbigNQ dataset.

        Ground Truth Used:
        - F1: all_short_answers (pooled from all interpretations + nq_answer)
        - D-F1: interpretation_answer_sets (one set per interpretation)
        - Retrieval: relevant_doc_titles
        """
        # Extract ground truth using standardized function
        gt = extract_ground_truth(reference_item)

        # 1. Answer Quality: F1 Score
        # Compare prediction against ALL valid short answers
        f1_score = compute_max_f1(prediction, gt.all_short_answers) if gt.all_short_answers else 0.0

        # 2. Disambiguation F1: Coverage of interpretations
        # Each interpretation is "covered" if F1 >= threshold with any answer in its set
        covered = 0
        for answer_set in gt.interpretation_answer_sets:
            max_f1 = compute_max_f1(prediction, answer_set)
            if max_f1 >= self.d_f1_threshold:
                covered += 1

        total_interp = gt.num_interpretations
        d_f1 = covered / total_interp if total_interp > 0 else 0.0

        # 3. Retrieval Quality
        ndcg = compute_ndcg_at_k(retrieved_docs, gt.relevant_doc_titles, k=self.k)
        recall = compute_recall_at_k(retrieved_docs, gt.relevant_doc_titles, k=self.k)

        # 4. Efficiency
        total_time = retrieval_time + generation_time

        return {
            'dataset': 'ambignq',
            'f1_score': f1_score,
            'd_f1': d_f1,
            'interpretations_covered': covered,
            'total_interpretations': total_interp,
            f'ndcg@{self.k}': ndcg,
            f'recall@{self.k}': recall,
            'num_relevant_docs': len(gt.relevant_doc_titles),
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'total_tokens': total_tokens,
            # Debug info for ground truth verification
            '_gt_info': {
                'num_short_answers': len(gt.all_short_answers),
                'num_interpretations': gt.num_interpretations,
                'sample_answers': gt.all_short_answers[:3] if gt.all_short_answers else []
            }
        }

    def _evaluate_asqa(
        self,
        prediction: str,
        retrieved_docs: List[Dict[str, Any]],
        reference_item: Dict[str, Any],
        retrieval_time: float,
        generation_time: float,
        total_tokens: int
    ) -> Dict[str, Any]:
        """
        Evaluate for ASQA dataset with ROUGE-L and DR-F1.

        Ground Truth Used:
        - F1: all_short_answers (pooled from all qa_pairs)
        - D-F1: qa_pairs (QA-based extraction using RoBERTa)
        - ROUGE-L: long_answer_references (from annotations)
        - DR-F1: sqrt(D-F1 * ROUGE-L)
        - Retrieval: relevant_doc_titles
        """
        # Extract ground truth using standardized function
        gt = extract_ground_truth(reference_item)

        # 1. Answer Quality: F1 Score (against short answers)
        f1_score = compute_max_f1(prediction, gt.all_short_answers) if gt.all_short_answers else 0.0

        # 2. Disambiguation F1 (coverage of interpretations)
        # Uses QA-based extraction for ASQA to avoid false positives
        if TRANSFORMERS_AVAILABLE and gt.qa_pairs:
            # QA-based evaluation: extract answers from prediction using RoBERTa
            evaluator = get_asqa_qa_evaluator()
            d_f1, covered, total_interp = evaluator.evaluate_disambiguation(
                prediction, gt.qa_pairs, threshold=self.d_f1_threshold
            )
        else:
            # Fallback: naive token overlap
            covered = 0
            for answer_set in gt.interpretation_answer_sets:
                max_f1 = compute_max_f1(prediction, answer_set)
                if max_f1 >= self.d_f1_threshold:
                    covered += 1
            total_interp = gt.num_interpretations
            d_f1 = covered / total_interp if total_interp > 0 else 0.0

        # 3. ROUGE-L (long-form answer quality)
        rouge_l = compute_rouge_l(prediction, gt.long_answer_references)

        # 4. DR-F1 (primary ASQA metric) = geometric mean
        dr_f1 = compute_dr_f1(d_f1, rouge_l)

        # 5. Retrieval Quality
        ndcg = compute_ndcg_at_k(retrieved_docs, gt.relevant_doc_titles, k=self.k)
        recall = compute_recall_at_k(retrieved_docs, gt.relevant_doc_titles, k=self.k)

        # 6. Efficiency
        total_time = retrieval_time + generation_time

        return {
            'dataset': 'asqa',
            'f1_score': f1_score,
            'd_f1': d_f1,
            'rouge_l': rouge_l,
            'dr_f1': dr_f1,  # Primary ASQA metric
            'interpretations_covered': covered,
            'total_interpretations': total_interp,
            f'ndcg@{self.k}': ndcg,
            f'recall@{self.k}': recall,
            'num_relevant_docs': len(gt.relevant_doc_titles),
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'total_tokens': total_tokens,
            # Debug info for ground truth verification
            '_gt_info': {
                'num_short_answers': len(gt.all_short_answers),
                'num_interpretations': gt.num_interpretations,
                'num_long_answers': len(gt.long_answer_references),
                'sample_answers': gt.all_short_answers[:3] if gt.all_short_answers else [],
                'qa_based_eval': TRANSFORMERS_AVAILABLE and bool(gt.qa_pairs)
            }
        }

    def evaluate_batch(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of results and compute aggregate statistics.

        Auto-detects dataset type from results and includes appropriate metrics.

        Args:
            results: List of result dictionaries from RAG pipeline

        Returns:
            Dictionary with aggregate metrics
        """
        metrics = {
            'f1_scores': [],
            'd_f1_scores': [],
            'rouge_l_scores': [],  # ASQA only
            'dr_f1_scores': [],    # ASQA only
            'ndcg_scores': [],
            'recall_scores': [],
            'retrieval_times': [],
            'generation_times': [],
            'total_times': [],
            'total_tokens': [],
            'interpretations_covered': [],
            'total_interpretations': []
        }

        # Detect dataset type from first result
        is_asqa = False
        if results:
            first_eval = results[0].get('evaluation', {})
            is_asqa = first_eval.get('dataset') == 'asqa' or 'rouge_l' in first_eval

        for result in results:
            eval_result = result.get('evaluation', {})

            metrics['f1_scores'].append(eval_result.get('f1_score', 0.0))
            metrics['d_f1_scores'].append(eval_result.get('d_f1', 0.0))
            metrics['ndcg_scores'].append(eval_result.get(f'ndcg@{self.k}', 0.0))
            metrics['recall_scores'].append(eval_result.get(f'recall@{self.k}', 0.0))
            metrics['retrieval_times'].append(eval_result.get('retrieval_time', 0.0))
            metrics['generation_times'].append(eval_result.get('generation_time', 0.0))
            metrics['total_times'].append(eval_result.get('total_time', 0.0))
            metrics['total_tokens'].append(eval_result.get('total_tokens', 0))
            metrics['interpretations_covered'].append(eval_result.get('interpretations_covered', 0))
            metrics['total_interpretations'].append(eval_result.get('total_interpretations', 0))

            # ASQA-specific metrics
            if is_asqa:
                metrics['rouge_l_scores'].append(eval_result.get('rouge_l', 0.0))
                metrics['dr_f1_scores'].append(eval_result.get('dr_f1', 0.0))

        # Compute aggregate statistics
        n = len(results)

        aggregate = {
            'num_examples': n,
            'dataset': 'asqa' if is_asqa else 'ambignq',

            # Answer quality
            'mean_f1': np.mean(metrics['f1_scores']) if metrics['f1_scores'] else 0.0,
            'median_f1': np.median(metrics['f1_scores']) if metrics['f1_scores'] else 0.0,
            'std_f1': np.std(metrics['f1_scores']) if metrics['f1_scores'] else 0.0,

            # Ambiguity handling
            'mean_d_f1': np.mean(metrics['d_f1_scores']) if metrics['d_f1_scores'] else 0.0,
            'median_d_f1': np.median(metrics['d_f1_scores']) if metrics['d_f1_scores'] else 0.0,
            'std_d_f1': np.std(metrics['d_f1_scores']) if metrics['d_f1_scores'] else 0.0,
            'total_interpretations_covered': sum(metrics['interpretations_covered']),
            'total_interpretations': sum(metrics['total_interpretations']),
            'coverage_rate': (
                sum(metrics['interpretations_covered']) / sum(metrics['total_interpretations'])
                if sum(metrics['total_interpretations']) > 0 else 0.0
            ),

            # Retrieval quality
            f'mean_ndcg@{self.k}': np.mean(metrics['ndcg_scores']) if metrics['ndcg_scores'] else 0.0,
            f'mean_recall@{self.k}': np.mean(metrics['recall_scores']) if metrics['recall_scores'] else 0.0,

            # Efficiency
            'mean_retrieval_time': np.mean(metrics['retrieval_times']) if metrics['retrieval_times'] else 0.0,
            'mean_generation_time': np.mean(metrics['generation_times']) if metrics['generation_times'] else 0.0,
            'mean_total_time': np.mean(metrics['total_times']) if metrics['total_times'] else 0.0,
            'mean_tokens_per_query': np.mean(metrics['total_tokens']) if metrics['total_tokens'] else 0.0,
            'total_tokens': sum(metrics['total_tokens']),

            # Percentiles
            'p50_latency': np.percentile(metrics['total_times'], 50) if metrics['total_times'] else 0.0,
            'p95_latency': np.percentile(metrics['total_times'], 95) if metrics['total_times'] else 0.0,
            'p99_latency': np.percentile(metrics['total_times'], 99) if metrics['total_times'] else 0.0
        }

        # Add ASQA-specific aggregate metrics
        if is_asqa:
            aggregate['mean_rouge_l'] = np.mean(metrics['rouge_l_scores']) if metrics['rouge_l_scores'] else 0.0
            aggregate['median_rouge_l'] = np.median(metrics['rouge_l_scores']) if metrics['rouge_l_scores'] else 0.0
            aggregate['std_rouge_l'] = np.std(metrics['rouge_l_scores']) if metrics['rouge_l_scores'] else 0.0
            aggregate['mean_dr_f1'] = np.mean(metrics['dr_f1_scores']) if metrics['dr_f1_scores'] else 0.0
            aggregate['median_dr_f1'] = np.median(metrics['dr_f1_scores']) if metrics['dr_f1_scores'] else 0.0
            aggregate['std_dr_f1'] = np.std(metrics['dr_f1_scores']) if metrics['dr_f1_scores'] else 0.0

        return aggregate

    def evaluate_results_post_generation(
        self,
        results: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of results AFTER generation is complete.

        This method is designed for the decoupled evaluation approach where
        retrieval and generation happen first, then evaluation runs in batch.
        This is more efficient when using heavy evaluation models like RoBERTa.

        Args:
            results: List of result dictionaries with:
                - generated_answer: The generated text
                - retrieved_docs: List of retrieved documents
                - reference_data: Reference data for evaluation
                - retrieval_time: Time spent on retrieval
                - generation_time: Time spent on generation
                - total_tokens: Tokens used
            show_progress: Whether to show a progress bar

        Returns:
            Same results list with 'evaluation' field added to each result
        """
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(results, desc="Evaluating results")
            except ImportError:
                iterator = results
        else:
            iterator = results

        for result in iterator:
            prediction = result.get('generated_answer', '')
            retrieved_docs = result.get('retrieved_docs', [])
            reference_item = result.get('reference_data', {})
            retrieval_time = result.get('retrieval_time', 0.0)
            generation_time = result.get('generation_time', 0.0)
            total_tokens = result.get('total_tokens', 0)

            evaluation = self.evaluate_single(
                prediction=prediction,
                retrieved_docs=retrieved_docs,
                reference_item=reference_item,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_tokens=total_tokens
            )

            result['evaluation'] = evaluation

        return results


def print_evaluation_report(metrics: Dict[str, Any]) -> None:
    """
    Pretty-print evaluation metrics. Supports both AmbigNQ and ASQA.

    Args:
        metrics: Dictionary of aggregate metrics
    """
    is_asqa = metrics.get('dataset') == 'asqa' or 'mean_rouge_l' in metrics

    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("="*70)

    dataset_name = "ASQA" if is_asqa else "AmbigNQ"
    print(f"\nDataset: {dataset_name} ({metrics['num_examples']} examples)")

    print("\n" + "-"*70)
    print("ANSWER QUALITY (F1 Score)")
    print("-"*70)
    print(f"Mean F1:       {metrics['mean_f1']:.4f}")
    print(f"Median F1:     {metrics['median_f1']:.4f}")
    print(f"Std Dev F1:    {metrics['std_f1']:.4f}")

    # ASQA-specific: ROUGE-L for long-form answers
    if is_asqa:
        print("\n" + "-"*70)
        print("LONG-FORM ANSWER QUALITY (ROUGE-L)")
        print("-"*70)
        print(f"Mean ROUGE-L:    {metrics.get('mean_rouge_l', 0.0):.4f}")
        print(f"Median ROUGE-L:  {metrics.get('median_rouge_l', 0.0):.4f}")
        print(f"Std Dev ROUGE-L: {metrics.get('std_rouge_l', 0.0):.4f}")

    print("\n" + "-"*70)
    print("AMBIGUITY HANDLING (Disambiguation F1)")
    print("-"*70)
    print(f"Mean D-F1:     {metrics['mean_d_f1']:.4f}")
    print(f"Median D-F1:   {metrics['median_d_f1']:.4f}")
    print(f"Std Dev D-F1:  {metrics['std_d_f1']:.4f}")
    print(f"Coverage:      {metrics['total_interpretations_covered']}/{metrics['total_interpretations']} "
          f"({metrics['coverage_rate']:.2%})")

    # ASQA-specific: DR-F1 (primary ASQA metric)
    if is_asqa:
        print("\n" + "-"*70)
        print("PRIMARY ASQA METRIC (DR-F1 = sqrt(D-F1 x ROUGE-L))")
        print("-"*70)
        print(f"Mean DR-F1:    {metrics.get('mean_dr_f1', 0.0):.4f}")
        print(f"Median DR-F1:  {metrics.get('median_dr_f1', 0.0):.4f}")
        print(f"Std Dev DR-F1: {metrics.get('std_dr_f1', 0.0):.4f}")

    k = 5  # Default k value
    if 'mean_ndcg@5' in metrics:
        k = 5
    elif 'mean_ndcg@10' in metrics:
        k = 10

    print("\n" + "-"*70)
    print(f"RETRIEVAL QUALITY (k={k})")
    print("-"*70)
    print(f"Mean nDCG@{k}:  {metrics.get(f'mean_ndcg@{k}', 0.0):.4f}")
    print(f"Mean Recall@{k}: {metrics.get(f'mean_recall@{k}', 0.0):.4f}")

    print("\n" + "-"*70)
    print("EFFICIENCY")
    print("-"*70)
    print(f"Mean Retrieval Time:  {metrics['mean_retrieval_time']:.3f}s")
    print(f"Mean Generation Time: {metrics['mean_generation_time']:.3f}s")
    print(f"Mean Total Time:      {metrics['mean_total_time']:.3f}s")
    print(f"P50 Latency:          {metrics['p50_latency']:.3f}s")
    print(f"P95 Latency:          {metrics['p95_latency']:.3f}s")
    print(f"P99 Latency:          {metrics['p99_latency']:.3f}s")
    print(f"Mean Tokens/Query:    {metrics['mean_tokens_per_query']:.0f}")
    print(f"Total Tokens:         {metrics['total_tokens']:,}")

    print("\n" + "="*70)


def validate_ground_truth(item: Dict[str, Any], verbose: bool = True) -> bool:
    """
    Validate that ground truth can be correctly extracted from a data item.

    Args:
        item: Raw data item from AmbigNQ or ASQA
        verbose: Print detailed validation info

    Returns:
        True if validation passes
    """
    try:
        gt = extract_ground_truth(item)

        if verbose:
            print(f"\n{'='*60}")
            print(f"GROUND TRUTH VALIDATION")
            print(f"{'='*60}")
            print(f"Dataset: {gt.dataset_type.value}")
            print(f"Question: {item.get('question', 'N/A')[:80]}...")
            print(f"\nGround Truth Extracted:")
            print(f"  - Short answers (F1): {len(gt.all_short_answers)} answers")
            print(f"    Sample: {gt.all_short_answers[:3]}")
            print(f"  - Interpretations (D-F1): {gt.num_interpretations}")
            for i, ans_set in enumerate(gt.interpretation_answer_sets[:3]):
                print(f"    [{i+1}] {ans_set}")
            print(f"  - Long answers (ROUGE-L): {len(gt.long_answer_references)}")
            if gt.long_answer_references:
                print(f"    Sample: {gt.long_answer_references[0][:100]}...")
            print(f"  - Relevant docs: {len(gt.relevant_doc_titles)}")
            print(f"    Sample: {list(gt.relevant_doc_titles)[:3]}")

        # Basic validation checks
        assert len(gt.all_short_answers) > 0, "No short answers found"
        assert gt.num_interpretations > 0, "No interpretations found"

        if gt.dataset_type == DatasetType.ASQA:
            assert len(gt.long_answer_references) > 0, "ASQA: No long answers found"
            assert len(gt.qa_pairs) > 0, "ASQA: No qa_pairs found"

        if verbose:
            print(f"\n✓ Validation PASSED")

        return True

    except Exception as e:
        if verbose:
            print(f"\n✗ Validation FAILED: {e}")
        return False


if __name__ == "__main__":
    print("="*70)
    print("EVALUATION MODULE TEST")
    print("="*70)

    # -------------------------------------------------------------------------
    # Test 1: Basic F1 Computation
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("TEST 1: F1 Score Computation")
    print("-"*70)

    pred = "The NBA introduced the 3-point line in 1979"
    gt = "1979"
    f1 = compute_f1(pred, gt)
    print(f"Prediction: {pred}")
    print(f"Ground Truth: {gt}")
    print(f"F1: {f1:.4f}")
    # F1 is ~0.286 because pred has 7 tokens, gt has 1, and 1 overlaps
    # precision=1/7, recall=1/1, F1=2*(1/7*1)/(1/7+1) = 2/7 / (8/7) = 2/8 = 0.25
    # Actually: 2*(0.143*1)/(0.143+1) = 0.286/(1.143) = 0.25... let me check
    assert f1 > 0.2, f"F1 should be > 0.2, got {f1}"
    print("✓ PASSED")

    # -------------------------------------------------------------------------
    # Test 2: AmbigNQ Ground Truth Extraction
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("TEST 2: AmbigNQ Ground Truth Extraction")
    print("-"*70)

    ambignq_item = {
        "question": "When was the nba 3 point line introduced?",
        "annotations": [
            {"type": "singleAnswer", "answer": ["1979-80 season", "June 1979"]},
            {"type": "singleAnswer", "answer": ["June 1979"]}
        ],
        "nq_answer": ["1979"],
        "viewed_doc_titles": ["Three-point field goal"],
        "nq_doc_title": "Three-point field goal"
    }

    gt = extract_ground_truth(ambignq_item)
    print(f"Dataset detected: {gt.dataset_type.value}")
    print(f"All short answers: {gt.all_short_answers}")
    print(f"Num interpretations: {gt.num_interpretations}")
    print(f"Interpretation sets: {gt.interpretation_answer_sets}")
    print(f"Relevant docs: {gt.relevant_doc_titles}")

    assert gt.dataset_type == DatasetType.AMBIGNQ
    assert len(gt.all_short_answers) >= 3  # 1979-80 season, June 1979, 1979
    assert gt.num_interpretations == 2
    assert "Three-point field goal" in gt.relevant_doc_titles
    print("✓ PASSED")

    # -------------------------------------------------------------------------
    # Test 3: ASQA Ground Truth Extraction
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("TEST 3: ASQA Ground Truth Extraction")
    print("-"*70)

    asqa_item = {
        "id": "test123",
        "question": "When was the first pirates movie released?",
        "dataset": "asqa",
        "qa_pairs": [
            {"question": "When premiered at Disneyland?", "short_answers": ["June 28, 2003"], "wikipage": "Pirates"},
            {"question": "When released in US?", "short_answers": ["July 9, 2003"], "wikipage": "Pirates US"}
        ],
        "annotations": [
            {"long_answer": "The first Pirates movie premiered on June 28, 2003 at Disneyland and was released in the US on July 9, 2003."}
        ],
        "wikipages": [{"title": "Pirates of the Caribbean"}]
    }

    gt = extract_ground_truth(asqa_item)
    print(f"Dataset detected: {gt.dataset_type.value}")
    print(f"All short answers: {gt.all_short_answers}")
    print(f"Num interpretations: {gt.num_interpretations}")
    print(f"QA pairs: {len(gt.qa_pairs)}")
    print(f"Long answers: {len(gt.long_answer_references)}")
    print(f"Relevant docs: {gt.relevant_doc_titles}")

    assert gt.dataset_type == DatasetType.ASQA
    assert "June 28, 2003" in gt.all_short_answers
    assert "July 9, 2003" in gt.all_short_answers
    assert gt.num_interpretations == 2
    assert len(gt.long_answer_references) == 1
    assert len(gt.qa_pairs) == 2
    print("✓ PASSED")

    # -------------------------------------------------------------------------
    # Test 4: D-F1 Computation
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("TEST 4: D-F1 Computation (Disambiguation Coverage)")
    print("-"*70)

    # Use answers that match the ground truth more directly
    pred = "June 1979"  # Exact match to both interpretation sets
    gt = extract_ground_truth(ambignq_item)

    covered = 0
    for answer_set in gt.interpretation_answer_sets:
        max_f1 = compute_max_f1(pred, answer_set)
        print(f"  Answer set {answer_set}: max F1 = {max_f1:.4f}")
        if max_f1 >= 0.5:
            covered += 1

    d_f1 = covered / gt.num_interpretations
    print(f"D-F1: {d_f1:.4f} ({covered}/{gt.num_interpretations} covered)")
    # Both interpretations contain "June 1979" so D-F1 should be 1.0
    assert d_f1 == 1.0, f"D-F1 should be 1.0, got {d_f1}"
    print("✓ PASSED")

    # -------------------------------------------------------------------------
    # Test 5: DR-F1 (Geometric Mean)
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("TEST 5: DR-F1 (Geometric Mean)")
    print("-"*70)

    dr_f1 = compute_dr_f1(0.8, 0.5)
    expected = math.sqrt(0.8 * 0.5)  # ~0.632
    print(f"D-F1=0.8, ROUGE-L=0.5 -> DR-F1={dr_f1:.4f}")
    print(f"Expected (geometric mean): {expected:.4f}")
    assert abs(dr_f1 - expected) < 0.001, f"DR-F1 should be {expected}, got {dr_f1}"
    print("✓ PASSED")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)
