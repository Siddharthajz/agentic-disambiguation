"""
Agentic Disambiguation - Novel Framework

Multi-agent RAG pipeline for handling ambiguous questions:
1. Sub-query decomposition: Break ambiguous question into specific sub-queries
2. HyDE retrieval: Generate hypothetical documents for each sub-query
3. Enhanced retrieval: Use HyDE docs to improve retrieval
4. Multi-answer generation: Generate answers for each interpretation
5. Answer synthesis: Combine answers to cover all interpretations

Uses LangGraph for agent orchestration and shared core components.
"""

import argparse
import asyncio
import json
import logging
import os
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
    HyDEGenerator,
    RAGResult,
    RetrievalResult
)
from evaluation import RAGEvaluator, print_evaluation_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgenticDisambiguation:
    """
    Agentic disambiguation framework for ambiguous questions.

    Pipeline stages:
    1. Ambiguity Detection: Determine if question is ambiguous
    2. Sub-query Generation: Decompose into specific interpretations
    3. HyDE Generation: Create hypothetical documents for each sub-query
    4. Enhanced Retrieval: Retrieve using both sub-queries and HyDE docs
    5. Answer Generation: Generate answer for each interpretation
    6. Answer Synthesis: Combine into comprehensive answer
    """

    def __init__(self, config: RAGConfig):
        """
        Initialize agentic disambiguation framework.

        Args:
            config: RAG configuration
        """
        self.config = config

        # Setup cache
        self.cache = RetrievalCache(
            cache_dir=config.cache_dir,
            enabled=config.use_cache
        ) if config.use_cache else None

        # Initialize retrievers (lazy loading)
        self.retriever = None

        # Initialize generators
        self.generator = OpenAIGenerator(
            model=config.llm_model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            api_key=config.openai_api_key
        )

        self.hyde_generator = HyDEGenerator(
            model=config.llm_model,
            max_tokens=config.max_tokens,
            temperature=0.7,  # Higher for diversity
            api_key=config.openai_api_key
        )

        # Initialize evaluator
        self.evaluator = RAGEvaluator(
            k=config.top_k,
            d_f1_threshold=config.d_f1_threshold
        )

        logger.info("Agentic Disambiguation initialized")

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

    async def _detect_ambiguity(self, question: str) -> Tuple[bool, float]:
        """
        Detect if question is ambiguous.

        Args:
            question: Question to analyze

        Returns:
            Tuple of (is_ambiguous, confidence_score)
        """
        # TODO: Implement LLM-based ambiguity detection
        # For now, assume all questions are potentially ambiguous
        # Future: Use LLM to classify and score ambiguity
        return True, 1.0

    async def _generate_subqueries(self, question: str) -> List[str]:
        """
        Decompose ambiguous question into specific sub-queries.

        Args:
            question: Ambiguous question

        Returns:
            List of specific sub-queries representing different interpretations
        """
        # TODO: Implement LLM-based sub-query generation
        # This is a placeholder - implement using OpenAI API
        prompt = f"""The following question is ambiguous and has multiple possible interpretations.
Generate 2-4 specific questions that represent different interpretations of the original question.

Original Question: {question}

Generate specific questions (one per line):"""

        # Placeholder: return original question
        # Future implementation:
        # 1. Use LLM to generate sub-queries
        # 2. Parse and validate sub-queries
        # 3. Return list of specific questions
        logger.warning("Sub-query generation not yet implemented - using original question")
        return [question]

    async def _generate_hyde_documents(
        self,
        subqueries: List[str]
    ) -> Dict[str, List[str]]:
        """
        Generate hypothetical documents for each sub-query using HyDE.

        Args:
            subqueries: List of sub-queries

        Returns:
            Dictionary mapping sub-query to list of hypothetical documents
        """
        hyde_docs = {}

        for subquery in subqueries:
            # Generate hypothetical document
            doc, gen_time, tokens = await self.hyde_generator.generate_hypothetical_document(subquery)
            hyde_docs[subquery] = [doc]
            logger.debug(f"Generated HyDE doc for: {subquery[:50]}...")

        return hyde_docs

    async def _retrieve_with_hyde(
        self,
        subquery: str,
        hyde_doc: str
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using both sub-query and HyDE document.

        Args:
            subquery: Specific sub-query
            hyde_doc: Hypothetical document

        Returns:
            List of retrieved documents
        """
        # Retrieve using sub-query
        subquery_results = self.retriever.retrieve(subquery, k=self.config.top_k)

        # Retrieve using HyDE document
        hyde_results = self.retriever.retrieve(hyde_doc, k=self.config.top_k)

        # Merge results (deduplicate by doc_id)
        doc_map = {}
        for doc in subquery_results + hyde_results:
            if doc.doc_id not in doc_map:
                doc_map[doc.doc_id] = doc

        # Re-rank by score
        merged_results = sorted(doc_map.values(), key=lambda x: x.score, reverse=True)

        return merged_results[:self.config.top_k]

    async def run_single(
        self,
        question: str,
        question_id: str,
        reference_data: Dict[str, Any]
    ) -> RAGResult:
        """
        Run agentic disambiguation pipeline on a single question.

        Args:
            question: Question to answer
            question_id: Unique question identifier
            reference_data: Reference data for evaluation

        Returns:
            RAG result with comprehensive answer
        """
        total_retrieval_time = 0.0
        total_generation_time = 0.0
        total_tokens = 0

        # Stage 1: Detect ambiguity
        is_ambiguous, ambiguity_score = await self._detect_ambiguity(question)

        # Stage 2: Generate sub-queries
        start_time = time.time()
        subqueries = await self._generate_subqueries(question)
        query_gen_time = time.time() - start_time
        logger.debug(f"Generated {len(subqueries)} sub-queries")

        # Stage 3: Generate HyDE documents
        start_time = time.time()
        hyde_docs = await self._generate_hyde_documents(subqueries)
        hyde_gen_time = time.time() - start_time
        total_generation_time += hyde_gen_time

        # Stage 4: Enhanced retrieval for each sub-query
        all_retrieved_docs = []
        for subquery in subqueries:
            start_time = time.time()
            hyde_doc = hyde_docs[subquery][0] if subquery in hyde_docs else subquery

            # Retrieve with HyDE
            docs = await self._retrieve_with_hyde(subquery, hyde_doc)
            retrieval_time = time.time() - start_time
            total_retrieval_time += retrieval_time

            all_retrieved_docs.extend(docs)

        # Deduplicate retrieved docs
        doc_map = {doc.doc_id: doc for doc in all_retrieved_docs}
        unique_docs = list(doc_map.values())[:self.config.top_k]

        # Stage 5: Generate comprehensive answer
        start_time = time.time()
        answer, gen_time, tokens = await self.generator.generate(question, unique_docs)
        total_generation_time += gen_time
        total_tokens += tokens

        # Evaluation
        evaluation = self.evaluator.evaluate_single(
            prediction=answer,
            retrieved_docs=[doc.to_dict() for doc in unique_docs],
            reference_item=reference_data,
            retrieval_time=total_retrieval_time,
            generation_time=total_generation_time,
            total_tokens=total_tokens
        )

        return RAGResult(
            question_id=question_id,
            question=question,
            retrieved_docs=[doc.to_dict() for doc in unique_docs],
            generated_answer=answer,
            reference_data=reference_data,
            retrieval_time=total_retrieval_time,
            generation_time=total_generation_time,
            total_tokens=total_tokens,
            evaluation=evaluation,
            metadata={
                "is_ambiguous": is_ambiguous,
                "ambiguity_score": ambiguity_score,
                "num_subqueries": len(subqueries),
                "subqueries": subqueries
            }
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
        Run agentic disambiguation on a batch of questions.

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
            desc=f"Agentic Disambiguation ({self.config.retrieval_mode})"
        )

        return results


def main():
    """Main entry point for agentic disambiguation."""
    parser = argparse.ArgumentParser(
        description="Agentic Disambiguation Framework (Novel Approach)"
    )

    # Data paths
    parser.add_argument("--data-path", type=str, default="data/ambignq_test.json")
    parser.add_argument("--output-path", type=str, default="results/agentic_disambiguation_results.json")

    # Retrieval settings
    parser.add_argument("--retrieval-mode", type=str, default="hybrid", choices=["sparse", "dense", "hybrid", "all"])
    parser.add_argument("--sparse-index", type=str, default="wikipedia-dpr")
    parser.add_argument("--dense-index", type=str, default="wikipedia-dpr-100w.bpr-single-nq")
    parser.add_argument("--dense-encoder", type=str, default="castorini/bpr-nq-question-encoder")
    parser.add_argument("--top-k", type=int, default=5)

    # Generation settings
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-tokens", type=int, default=200)

    # Performance settings
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--no-cache", action="store_false", dest="use_cache")

    # Experiment settings
    parser.add_argument("--limit", type=int, default=None)

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

        # Initialize agentic framework
        framework = AgenticDisambiguation(config)
        framework._load_retriever(mode)

        # Run experiments
        results = asyncio.run(framework.run_batch(test_data, limit=args.limit))

        # Compute metrics
        logger.info(f"\nComputing aggregate metrics for {mode}...")
        aggregate_metrics = framework.evaluator.evaluate_batch([r.to_dict() for r in results])

        # Print report
        logger.info(f"\n{'='*60}\n  Results for {mode.upper()} mode\n{'='*60}")
        print_evaluation_report(aggregate_metrics)

        # Save results
        all_results[mode] = {
            "config": config.to_dict(),
            "aggregate_metrics": aggregate_metrics,
            "results": [r.to_dict() for r in results]
        }

        # Cleanup
        if args.retrieval_mode == "all":
            framework._cleanup_retriever()

        # Cache stats
        if framework.cache:
            cache_stats = framework.cache.get_stats()
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

    logger.info("\n" + "="*60)
    logger.info("NOTE: This is a skeleton implementation.")
    logger.info("TODO: Implement sub-query generation and LangGraph orchestration")
    logger.info("="*60)


if __name__ == "__main__":
    main()
