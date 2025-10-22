"""
Vanilla RAG Baseline - Optimized with Modular Architecture

Standard RAG pipeline: Retrieve → Generate
Uses shared core components for retrieval and generation.
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import gc

from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv

from core import (
    RAGConfig,
    RetrievalCache,
    create_retriever,
    OpenAIGenerator,
    RAGResult
)
from evaluation import RAGEvaluator, print_evaluation_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VanillaRAG:
    """
    Vanilla RAG implementation using modular components.

    Pipeline: Retrieve relevant documents → Generate answer
    """

    def __init__(self, config: RAGConfig):
        """
        Initialize vanilla RAG pipeline.

        Args:
            config: RAG configuration
        """
        self.config = config

        # Setup cache
        self.cache = RetrievalCache(
            cache_dir=config.cache_dir,
            enabled=config.use_cache
        ) if config.use_cache else None

        # Initialize retriever (lazy loading based on mode)
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

        logger.info("Vanilla RAG initialized")

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

    async def run_single(
        self,
        question: str,
        question_id: str,
        reference_data: Dict[str, Any]
    ) -> RAGResult:
        """
        Run RAG pipeline on a single question.

        Args:
            question: Question to answer
            question_id: Unique question identifier
            reference_data: Reference data for evaluation

        Returns:
            RAG result with answer and metrics
        """
        # Retrieval
        start_time = time.time()
        retrieved_docs = self.retriever.retrieve(question)
        retrieval_time = time.time() - start_time

        # Generation
        answer, generation_time, total_tokens = await self.generator.generate(
            question, retrieved_docs
        )

        # Evaluation
        evaluation = self.evaluator.evaluate_single(
            prediction=answer,
            retrieved_docs=[doc.to_dict() for doc in retrieved_docs],
            reference_item=reference_data,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_tokens=total_tokens
        )

        return RAGResult(
            question_id=question_id,
            question=question,
            retrieved_docs=[doc.to_dict() for doc in retrieved_docs],
            generated_answer=answer,
            reference_data=reference_data,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_tokens=total_tokens,
            evaluation=evaluation
        )

    async def _process_with_semaphore(
        self,
        item: Dict[str, Any],
        semaphore: asyncio.Semaphore
    ) -> RAGResult:
        """Process a single item with concurrency control."""
        question = item['question']
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
        Run RAG pipeline on a batch of questions.

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
            desc=f"Vanilla RAG ({self.config.retrieval_mode}, concurrency={self.config.concurrency})"
        )

        return results


def main():
    """Main entry point for vanilla RAG baseline."""
    parser = argparse.ArgumentParser(
        description="Vanilla RAG Baseline with optimized modular architecture"
    )

    # Data paths
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ambignq_test.json",
        help="Path to AmbigNQ test data"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/vanilla_rag_results.json",
        help="Path to save results"
    )

    # Retrieval settings
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="all",
        choices=["sparse", "dense", "hybrid", "all"],
        help="Retrieval mode(s) to run"
    )
    parser.add_argument(
        "--sparse-index",
        type=str,
        default="wikipedia-dpr",
        help="PySerini sparse index name"
    )
    parser.add_argument(
        "--dense-index",
        type=str,
        default="ambigqa_wiki.index",
        help="Path to FAISS index file"
    )
    parser.add_argument(
        "--dense-encoder",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name for query encoding"
    )
    parser.add_argument(
        "--dense-metadata",
        type=str,
        default="ambigqa_wiki_metadata.json",
        help="Path to metadata JSON file for dense retrieval"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )

    # Generation settings
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Max tokens for generation"
    )

    # Performance settings
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent OpenAI requests"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Enable retrieval caching"
    )
    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Disable retrieval caching"
    )

    # Experiment settings
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test examples"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Load test data
    logger.info(f"Loading test data from {args.data_path}...")
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

        # Initialize RAG
        rag = VanillaRAG(config)

        # Load retriever for this mode
        rag._load_retriever(mode)

        # Run experiments
        results = asyncio.run(rag.run_batch(test_data, limit=args.limit))

        # Compute aggregate metrics
        logger.info(f"\nComputing aggregate metrics for {mode}...")
        aggregate_metrics = rag.evaluator.evaluate_batch(
            [r.to_dict() for r in results]
        )

        # Print report
        logger.info(f"\n{'='*60}\n  Results for {mode.upper()} mode\n{'='*60}")
        print_evaluation_report(aggregate_metrics)

        # Save results
        all_results[mode] = {
            "config": config.to_dict(),
            "aggregate_metrics": aggregate_metrics,
            "results": [r.to_dict() for r in results]
        }

        # Cleanup if running multiple modes
        if args.retrieval_mode == "all":
            rag._cleanup_retriever()

        # Print cache stats
        if rag.cache:
            cache_stats = rag.cache.get_stats()
            logger.info(f"\nCache stats: {cache_stats}")

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.retrieval_mode == "all":
        for mode, data in all_results.items():
            mode_output = output_path.parent / f"{output_path.stem}_{mode}{output_path.suffix}"
            with open(mode_output, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"\n{mode.upper()} results saved to {mode_output}")
    else:
        with open(output_path, 'w') as f:
            json.dump(all_results[args.retrieval_mode], f, indent=2)
        logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
