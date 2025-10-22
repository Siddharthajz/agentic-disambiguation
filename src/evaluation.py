import re
import string
from collections import Counter
from typing import List, Dict, Any, Set, Tuple
import numpy as np


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


class RAGEvaluator:
    """Comprehensive evaluator for RAG systems on AmbigNQ."""

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
        Evaluate a single prediction.

        Args:
            prediction: Generated answer
            retrieved_docs: Retrieved documents
            reference_item: Reference data from AmbigNQ
            retrieval_time: Time spent on retrieval (seconds)
            generation_time: Time spent on generation (seconds)
            total_tokens: Total tokens processed

        Returns:
            Dictionary of metrics
        """
        # Extract reference data
        annotations = reference_item.get('annotations', [])
        nq_answer = reference_item.get('nq_answer', [])

        # 1. Answer Quality: F1 Score
        # Compute F1 against all possible answers
        all_reference_answers = []
        answer_sets = extract_answers_from_annotations(annotations)
        for answer_set in answer_sets:
            all_reference_answers.extend(answer_set)

        # Also include NQ answers
        if isinstance(nq_answer, str):
            all_reference_answers.append(nq_answer)
        elif isinstance(nq_answer, list):
            all_reference_answers.extend(nq_answer)

        f1_score = compute_max_f1(prediction, all_reference_answers) if all_reference_answers else 0.0

        # 2. Ambiguity Handling: D-F1
        d_f1, covered, total_interp = compute_disambiguation_f1(
            prediction,
            annotations,
            threshold=self.d_f1_threshold
        )

        # 3. Retrieval Quality
        relevant_titles = extract_relevant_doc_titles(reference_item)
        ndcg = compute_ndcg_at_k(retrieved_docs, relevant_titles, k=self.k)
        recall = compute_recall_at_k(retrieved_docs, relevant_titles, k=self.k)

        # 4. Efficiency
        total_time = retrieval_time + generation_time

        return {
            # Answer quality
            'f1_score': f1_score,

            # Ambiguity handling
            'd_f1': d_f1,
            'interpretations_covered': covered,
            'total_interpretations': total_interp,

            # Retrieval quality
            f'ndcg@{self.k}': ndcg,
            f'recall@{self.k}': recall,
            'num_relevant_docs': len(relevant_titles),

            # Efficiency
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'total_tokens': total_tokens
        }

    def evaluate_batch(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of results and compute aggregate statistics.

        Args:
            results: List of result dictionaries from RAG pipeline

        Returns:
            Dictionary with aggregate metrics
        """
        metrics = {
            'f1_scores': [],
            'd_f1_scores': [],
            'ndcg_scores': [],
            'recall_scores': [],
            'retrieval_times': [],
            'generation_times': [],
            'total_times': [],
            'total_tokens': [],
            'interpretations_covered': [],
            'total_interpretations': []
        }

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

        # Compute aggregate statistics
        n = len(results)

        aggregate = {
            'num_examples': n,

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

        return aggregate


def print_evaluation_report(metrics: Dict[str, Any]) -> None:
    """
    Pretty-print evaluation metrics.

    Args:
        metrics: Dictionary of aggregate metrics
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("="*70)

    print(f"\nDataset: {metrics['num_examples']} examples")

    print("\n" + "-"*70)
    print("ANSWER QUALITY (F1 Score)")
    print("-"*70)
    print(f"Mean F1:       {metrics['mean_f1']:.4f}")
    print(f"Median F1:     {metrics['median_f1']:.4f}")
    print(f"Std Dev F1:    {metrics['std_f1']:.4f}")

    print("\n" + "-"*70)
    print("AMBIGUITY HANDLING (Disambiguation F1)")
    print("-"*70)
    print(f"Mean D-F1:     {metrics['mean_d_f1']:.4f}")
    print(f"Median D-F1:   {metrics['median_d_f1']:.4f}")
    print(f"Std Dev D-F1:  {metrics['std_d_f1']:.4f}")
    print(f"Coverage:      {metrics['total_interpretations_covered']}/{metrics['total_interpretations']} "
          f"({metrics['coverage_rate']:.2%})")

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


if __name__ == "__main__":
    # Example usage
    print("Evaluation Module Test")

    # Test F1 computation
    pred = "The NBA introduced the 3-point line in 1979"
    gt = "1979"
    f1 = compute_f1(pred, gt)
    print(f"\nF1 Score Test:")
    print(f"Prediction: {pred}")
    print(f"Ground Truth: {gt}")
    print(f"F1: {f1:.4f}")

    # Test D-F1
    annotations = [
        {"type": "singleAnswer", "answer": ["1979", "1979-80 season"]},
        {"type": "singleAnswer", "answer": ["June 1979"]}
    ]
    d_f1, covered, total = compute_disambiguation_f1(pred, annotations)
    print(f"\nD-F1 Test:")
    print(f"D-F1: {d_f1:.4f} ({covered}/{total} interpretations covered)")
