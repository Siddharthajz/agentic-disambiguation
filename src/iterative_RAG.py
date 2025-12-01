"""
Iterative RAG Baseline

Multi-round RAG pipeline with iterative refinement:
1. Initial retrieval and generation
2. Check answer quality
3. If needed, reformulate query and retrieve again
4. Generate refined answer

Uses shared core components for retrieval and generation.
"""

# Fix for Java/OpenMP conflict on Apple Silicon
# Must be set BEFORE importing transformers/torch
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import gc

from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv

from core import (
    RAGConfig,
    RetrievalCache,
    create_retriever,
    OpenAIGenerator,
    RAGResult,
    RetrievalResult,
    get_model_name_from_config,
    get_organized_output_path,
    ensure_output_directory,
    load_existing_results,
    get_processed_question_ids,
    filter_unprocessed_data,
    merge_results,
    detect_dataset_from_item,
    get_question_field,
)
from evaluation import RAGEvaluator, print_evaluation_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IterativeRAG:
    """
    Iterative RAG implementation with multi-round refinement.

    Pipeline:
    1. Retrieve → Generate initial answer
    2. If confidence is low or answer incomplete, iterate:
       a. Reformulate query based on initial answer
       b. Retrieve new documents
       c. Generate refined answer
    3. Return best answer after max iterations
    """

    def __init__(self, config: RAGConfig, max_iterations: int = 3, dataset: str = "ambignq"):
        """
        Initialize iterative RAG pipeline.

        Args:
            config: RAG configuration
            max_iterations: Maximum number of iterations
            dataset: Dataset type ("ambignq" or "asqa")
        """
        self.config = config
        self.max_iterations = max_iterations
        self.dataset = dataset

        # Setup cache - only create if explicitly enabled
        if config.use_cache:
            self.cache = RetrievalCache(
                cache_dir=config.cache_dir,
                enabled=True
            )
            logger.info("Retrieval cache ENABLED")
        else:
            self.cache = None
            logger.info("Retrieval cache DISABLED - all retrievals will be fresh")

        # Initialize retriever (lazy loading)
        self.retriever = None

        # Initialize generator
        self.generator = OpenAIGenerator(
            model=config.llm_model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            api_key=config.openai_api_key
        )

        # Initialize evaluator
        self.evaluator = RAGEvaluator(
            k=config.top_k,
            d_f1_threshold=config.d_f1_threshold
        )

        logger.info(f"Iterative RAG initialized (max_iterations={max_iterations})")

    def _load_retriever(self, mode: str):
        """Load retriever for specified mode."""
        if self.retriever is not None:
            return

        logger.info(f"Loading retriever: {mode}")
        self.retriever = create_retriever(
            mode=mode,
            sparse_index=self.config.sparse_index,
            dense_index=self.config.dense_index,
            dense_encoder=self.config.dense_encoder,
            dense_metadata=self.config.dense_metadata,
            top_k=self.config.top_k,
            cache=self.cache
        )

    def _cleanup_retriever(self):
        """Clean up retriever to free memory."""
        if self.retriever is not None:
            logger.info("Cleaning up retriever...")
            del self.retriever
            self.retriever = None
            gc.collect()

    async def _reformulate_query(
        self,
        original_question: str,
        previous_answer: str,
        iteration: int
    ) -> str:
        """
        Reformulate query based on previous answer using LLM.

        Args:
            original_question: Original question
            previous_answer: Previously generated answer
            iteration: Current iteration number

        Returns:
            Reformulated query
        """
        # Use LLM to reformulate the query based on what's missing
        reformulation_prompt = f"""You are a query reformulation expert. Given an original question and a partial answer, generate a more specific search query to find additional missing information.

Original Question: {original_question}

Current Answer: {previous_answer}

Analyze what information might be missing or incomplete in the current answer. Generate a focused search query that would help retrieve documents containing the missing information. The query should:
1. Be specific and targeted
2. Focus on gaps in the current answer
3. Be optimized for document retrieval
4. Be concise (1-2 sentences max)

Output ONLY the reformulated search query, nothing else."""

        try:
            # Generate reformulated query using direct API call
            response = await self.generator.client.chat.completions.create(
                model=self.generator.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that reformulates search queries to find missing information. Respond with only the reformulated query."
                    },
                    {
                        "role": "user",
                        "content": reformulation_prompt
                    }
                ],
                max_tokens=100,
                temperature=0.3
            )

            reformulated_query = response.choices[0].message.content.strip()

            # Clean up the reformulated query - remove any quotation marks or extra formatting
            reformulated_query = reformulated_query.strip('"').strip("'").strip()

            # If reformulation failed or is too short, fall back to original
            if not reformulated_query or len(reformulated_query) < 10:
                logger.warning(f"Query reformulation produced short result, using original question")
                return original_question

            logger.info(f"Reformulated query (iteration {iteration}): {reformulated_query}")
            return reformulated_query

        except Exception as e:
            logger.error(f"Error in query reformulation: {e}")
            # Fall back to original question on error
            return original_question

    def _calculate_progress_score(
        self,
        retrieved_docs: List[RetrievalResult],
        all_previous_docs: List[RetrievalResult]
    ) -> float:
        """
        Calculate progress score based on retrieval quality.

        Inspired by IM-RAG's Progress Tracker, this measures how much new relevant
        information was retrieved in the current iteration.

        Args:
            retrieved_docs: Documents retrieved in current iteration
            all_previous_docs: All documents from previous iterations

        Returns:
            Progress score between 0 and 1 (higher = more progress)
        """
        if not retrieved_docs:
            return 0.0

        # Calculate average retrieval score
        avg_score = sum(doc.score for doc in retrieved_docs[:5]) / min(5, len(retrieved_docs))

        # Calculate novelty - how many new documents were found
        previous_doc_ids = {doc.doc_id for doc in all_previous_docs}
        new_docs = [doc for doc in retrieved_docs if doc.doc_id not in previous_doc_ids]
        novelty_ratio = len(new_docs) / len(retrieved_docs) if retrieved_docs else 0.0

        # Combined progress score (weighted average)
        progress_score = 0.6 * avg_score + 0.4 * novelty_ratio

        return progress_score

    async def _should_iterate(
        self,
        answer: str,
        iteration: int,
        retrieved_docs: List[RetrievalResult],
        all_previous_docs: List[RetrievalResult] = None,
        progress_threshold: float = 0.3
    ) -> bool:
        """
        Determine if another iteration is needed based on answer quality and confidence.

        Uses multiple heuristics inspired by iterative RAG papers:
        - Answer completeness (length, uncertainty markers)
        - Retrieval progress score (quality + novelty)
        - Maximum iteration limit

        Args:
            answer: Current answer
            iteration: Current iteration number
            retrieved_docs: Retrieved documents in this iteration
            all_previous_docs: All documents from previous iterations
            progress_threshold: Minimum progress score to continue (default 0.3)

        Returns:
            True if should continue iterating
        """
        # Stop if max iterations reached
        if iteration >= self.max_iterations:
            logger.debug(f"Max iterations ({self.max_iterations}) reached, stopping")
            return False

        # Check for empty or very short answers (likely incomplete)
        if len(answer.strip()) < 20:
            logger.debug(f"Answer too short ({len(answer)} chars), continuing iteration")
            return True

        # Check for uncertainty markers indicating incomplete information
        uncertainty_markers = [
            "unclear",
            "unknown",
            "not sure",
            "don't know",
            "cannot determine",
            "insufficient information",
            "more information needed",
            "ambiguous",
            "uncertain",
            "possibly",
            "might be",
            "could be",
            "no clear answer",
            "not enough information"
        ]

        answer_lower = answer.lower()
        for marker in uncertainty_markers:
            if marker in answer_lower:
                logger.debug(f"Uncertainty marker '{marker}' found in answer, continuing iteration")
                return True

        # Calculate progress score based on retrieval quality and novelty
        if all_previous_docs is not None:
            progress_score = self._calculate_progress_score(retrieved_docs, all_previous_docs)
            logger.debug(f"Progress score: {progress_score:.3f} (threshold: {progress_threshold})")

            # If progress is very low, retrieval isn't helping - stop iterating
            if progress_score < progress_threshold and iteration >= 2:
                logger.debug(f"Low progress score ({progress_score:.3f}), stopping iteration")
                return False

        # Check retrieval quality - if scores are very low, might need better query
        if retrieved_docs:
            avg_score = sum(doc.score for doc in retrieved_docs[:5]) / min(5, len(retrieved_docs))
            # If average score is very low, the retrieval might not be good
            if avg_score < 0.5 and iteration < 2:
                logger.debug(f"Low retrieval scores (avg={avg_score:.3f}), continuing iteration")
                return True

        # Check answer length - very long answers might be complete
        # Short/medium answers might benefit from more context
        if len(answer.split()) < 30 and iteration < 2:
            logger.debug(f"Answer relatively short ({len(answer.split())} words), continuing iteration")
            return True

        # If we've done at least one iteration and answer seems complete, stop
        if iteration >= 1:
            logger.debug(f"Answer appears complete after {iteration} iteration(s), stopping")
            return False

        # Default: iterate at least once
        return True

    async def run_single(
        self,
        question: str,
        question_id: str,
        reference_data: Dict[str, Any]
    ) -> RAGResult:
        """
        Run iterative RAG pipeline on a single question.

        Args:
            question: Question to answer
            question_id: Unique question identifier
            reference_data: Reference data for evaluation

        Returns:
            RAG result with best answer and metrics
        """
        total_retrieval_time = 0.0
        total_generation_time = 0.0
        total_tokens = 0

        current_query = question
        best_answer = ""
        all_retrieved_docs = []

        # Iterative refinement loop
        for iteration in range(1, self.max_iterations + 1):
            logger.debug(f"Iteration {iteration} for question: {question[:50]}...")

            # Retrieval
            start_time = time.time()
            retrieved_docs = self.retriever.retrieve(current_query)
            retrieval_time = time.time() - start_time
            total_retrieval_time += retrieval_time

            # Merge with previous docs (avoid duplicates)
            doc_ids = {doc.doc_id for doc in all_retrieved_docs}
            for doc in retrieved_docs:
                if doc.doc_id not in doc_ids:
                    all_retrieved_docs.append(doc)
                    doc_ids.add(doc.doc_id)
            sorted_docs = sorted(all_retrieved_docs, key=lambda d: d.score, reverse=True)
            top_docs = sorted_docs[:self.config.top_k]

            # Generation (with dataset-specific prompts)
            answer, generation_time, tokens = await self.generator.generate(
                question, top_docs, dataset=self.dataset
            )
            total_generation_time += generation_time
            total_tokens += tokens

            best_answer = answer

            # Check if should iterate (pass previous docs for progress tracking)
            previous_docs = all_retrieved_docs[:-len(retrieved_docs)] if len(all_retrieved_docs) > len(retrieved_docs) else []
            should_continue = await self._should_iterate(
                answer,
                iteration,
                retrieved_docs,
                all_previous_docs=previous_docs
            )
            if not should_continue:
                break

            # Reformulate query for next iteration
            current_query = await self._reformulate_query(question, answer, iteration)

        # Evaluation
        evaluation = self.evaluator.evaluate_single(
            prediction=best_answer,
            retrieved_docs=[doc.to_dict() for doc in all_retrieved_docs[:self.config.top_k]],
            reference_item=reference_data,
            retrieval_time=total_retrieval_time,
            generation_time=total_generation_time,
            total_tokens=total_tokens
        )

        return RAGResult(
            question_id=question_id,
            question=question,
            retrieved_docs=[doc.to_dict() for doc in all_retrieved_docs[:self.config.top_k]],
            generated_answer=best_answer,
            reference_data=reference_data,
            retrieval_time=total_retrieval_time,
            generation_time=total_generation_time,
            total_tokens=total_tokens,
            evaluation=evaluation,
            metadata={"num_iterations": iteration, "total_docs_retrieved": len(all_retrieved_docs)}
        )

    async def _process_with_semaphore(
        self,
        item: Dict[str, Any],
        semaphore: asyncio.Semaphore
    ) -> RAGResult:
        """Process a single item with concurrency control."""
        question = get_question_field(item, self.dataset)
        question_id = item.get('id', str(hash(question)))

        async with semaphore:
            try:
                return await self.run_single(question, question_id, item)
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                return RAGResult(
                    question_id=question_id,
                    question=question,
                    retrieved_docs=[],
                    generated_answer=f"ERROR: {str(e)}",
                    reference_data=item,
                    retrieval_time=0.0,
                    generation_time=0.0,
                    total_tokens=0,
                    evaluation={}
                )

    async def run_batch(
        self,
        test_data: List[Dict[str, Any]],
        limit: Optional[int] = None
    ) -> List[RAGResult]:
        """
        Run iterative RAG pipeline on a batch of questions.

        Args:
            test_data: List of test examples
            limit: Optional limit on number of examples

        Returns:
            List of RAG results
        """
        if limit:
            test_data = test_data[:limit]

        semaphore = asyncio.Semaphore(self.config.concurrency)
        tasks = [
            self._process_with_semaphore(item, semaphore)
            for item in test_data
        ]

        results = await async_tqdm.gather(
            *tasks,
            desc=f"Iterative RAG ({self.config.retrieval_mode}, max_iter={self.max_iterations})"
        )

        return results


def main():
    """Main entry point for iterative RAG baseline."""
    parser = argparse.ArgumentParser(
        description="Iterative RAG Baseline with multi-round refinement"
    )

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="ambignq", choices=["ambignq", "asqa"], help="Dataset type")

    # Data paths
    parser.add_argument("--data-path", type=str, default=None, help="Path to test data (default: auto-select based on dataset)")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save results (default: organized by approach/mode/model)")

    # Retrieval settings
    parser.add_argument("--retrieval-mode", type=str, default="sparse", choices=["sparse", "dense", "hybrid", "all"])
    parser.add_argument("--sparse-index", type=str, default="wikipedia-dpr")
    parser.add_argument("--dense-index", type=str, default="../data/ambigqa_wiki.index")
    parser.add_argument("--dense-encoder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--dense-metadata", type=str, default="../data/ambigqa_wiki_metadata.json")
    parser.add_argument("--top-k", type=int, default=5)

    # Generation settings
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-tokens", type=int, default=200)

    # Iterative RAG specific
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum number of refinement iterations")

    # Performance settings
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent API requests (reduce if hitting rate limits)")
    parser.add_argument("--use-cache", action="store_true", default=False, help="Enable retrieval caching (NOT recommended for benchmarking)")
    parser.add_argument("--no-cache", action="store_false", dest="use_cache", help="Disable retrieval caching (recommended for accurate timing)")

    # Experiment settings
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Set default data path based on dataset
    if args.data_path is None:
        if args.dataset == "asqa":
            args.data_path = "../data/asqa_test.json"
        else:
            args.data_path = "../data/ambignq_test.json"

    # Load test data
    logger.info(f"Loading {args.dataset.upper()} test data from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        test_data = json.load(f)
    logger.info(f"Loaded {len(test_data)} test examples")

    # Determine modes to run
    modes_to_run = ["sparse", "dense", "hybrid"] if args.retrieval_mode == "all" else [args.retrieval_mode]
    all_results = {}

    for mode in modes_to_run:
        logger.info(f"\n{'='*60}\n  Running {mode.upper()} retrieval mode\n{'='*60}\n")

        # Create config
        config = RAGConfig.from_args(args)
        config.retrieval_mode = mode
        config.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Initialize iterative RAG
        rag = IterativeRAG(config, max_iterations=args.max_iterations, dataset=args.dataset)
        rag._load_retriever(mode)

        # Determine output path for resume capability
        if args.output_path is None:
            model_name = get_model_name_from_config(config.to_dict())
            is_test = args.limit is not None and args.limit < 100
            output_path = get_organized_output_path(
                approach="iterative",
                retrieval_mode=mode,
                model_name=model_name,
                is_test=is_test, 
                results_dir="../results"
            )
        else:
            output_path = Path(args.output_path)
            if args.retrieval_mode == "all":
                output_path = output_path.parent / f"{output_path.stem}_{mode}{output_path.suffix}"
        
        # Check for existing results and filter test data
        existing_results = load_existing_results(output_path)
        if existing_results:
            processed_ids = get_processed_question_ids(existing_results)
            original_count = len(test_data)
            test_data = filter_unprocessed_data(test_data, processed_ids)
            if len(test_data) < original_count:
                logger.info(f"Resuming: {len(processed_ids)} already processed, {len(test_data)} remaining")
            else:
                logger.info(f"Existing results found but no overlap, starting fresh")
        else:
            logger.info(f"Starting new experiment")

        # Run experiments (only on unprocessed data)
        if test_data:
            results = asyncio.run(rag.run_batch(test_data, limit=args.limit))
            new_results = [r.to_dict() for r in results]
        else:
            logger.info(f"All items already processed, skipping run")
            new_results = []

        # Merge with existing results
        config_dict = {**config.to_dict(), "max_iterations": args.max_iterations}
        merged_data = merge_results(existing_results, new_results, config_dict)

        # Recompute aggregate metrics on merged results
        logger.info(f"\nComputing aggregate metrics for {mode}...")
        aggregate_metrics = rag.evaluator.evaluate_batch(merged_data["results"])
        merged_data["aggregate_metrics"] = aggregate_metrics

        # Print report
        logger.info(f"\n{'='*60}\n  Results for {mode.upper()} mode\n{'='*60}")
        print_evaluation_report(aggregate_metrics)

        # Prepare results data
        all_results[mode] = merged_data

        # Save results immediately after each mode completes
        ensure_output_directory(output_path)
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
        logger.info(f"\n✓ {mode.upper() if args.retrieval_mode == 'all' else ''} Results saved to {output_path}")

        # Cleanup
        if args.retrieval_mode == "all":
            rag._cleanup_retriever()

        # Cache stats
        if rag.cache:
            cache_stats = rag.cache.get_stats()
            logger.info(f"\nCache stats: {cache_stats}")

    # Final summary if running all modes
    if args.retrieval_mode == "all":
        logger.info(f"\n{'='*60}\n  ALL MODES COMPLETED\n{'='*60}")
        logger.info(f"\nAll results saved in: {output_path.parent}")
        for mode in modes_to_run:
            mode_output = output_path.parent / f"{output_path.stem}_{mode}{output_path.suffix}"
            logger.info(f"  - {mode_output.name}")


if __name__ == "__main__":
    main()
