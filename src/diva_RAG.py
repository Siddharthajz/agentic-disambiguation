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
    RAGResult,
    RetrievalResult,
    get_model_name_from_config,
    get_organized_output_path,
    ensure_output_directory,
    load_existing_results,
    get_processed_question_ids,
    filter_unprocessed_data,
    merge_results,
    get_question_field
)
from core.generators import LlamaCppGenerator
from evaluation import RAGEvaluator, print_evaluation_report, detect_dataset_type, DatasetType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DIVARAG:
    """
    DIVA (Diversify–Verify–Adapt) RAG implementation.
    
    Pipeline:
    1. Diversify: Generate multiple interpretations of ambiguous question
    2. Retrieve: Get diverse passages for each interpretation
    3. Verify: Assess quality of retrieved passages
    4. Adapt: Generate answer using adaptive strategy based on verification
    """

    def __init__(
        self,
        config: RAGConfig,
        num_interpretations: int = 3,
        use_local_llm: bool = False,
        local_model_path: str = None,
        local_context_size: int = 4096,
        local_gpu_layers: int = -1,
        skip_verification: bool = False,
        dataset: str = "ambignq"
    ):
        """
        Initialize DIVA RAG pipeline.
        
        Args:
            config: RAG configuration
            num_interpretations: Number of question interpretations to generate
            use_local_llm: Whether to use local LLM instead of OpenAI
            local_model_path: Path to local GGUF model file
            local_context_size: Context window size for local LLM
            local_gpu_layers: GPU layers to offload (-1 for all)
            skip_verification: Skip verification step to save API calls
            dataset: Dataset type ("ambignq" or "asqa")
        """
        self.config = config
        self.num_interpretations = num_interpretations
        self.skip_verification = skip_verification
        self.dataset = dataset

        # Setup cache
        self.cache = RetrievalCache(
            cache_dir=config.cache_dir,
            enabled=config.use_cache
        ) if config.use_cache else None

        # Initialize retriever (lazy loading)
        self.retriever = None

        # Initialize generator
        if use_local_llm:
            if local_model_path is None:
                local_model_path = "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
            logger.info(f"Using local LLM: {local_model_path}")
            self.generator = LlamaCppGenerator(
                model_path=local_model_path,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                n_ctx=local_context_size,
                n_gpu_layers=local_gpu_layers
            )
        else:
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

        logger.info(f"DIVA RAG initialized (num_interpretations={num_interpretations})")

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

    async def _diversify(
        self,
        question: str
    ) -> List[str]:
        """
        Stage 1: Diversify - Generate multiple interpretations of ambiguous question.
        
        Args:
            question: Original ambiguous question
            
        Returns:
            List of question interpretations
        """
        diversify_prompt = f"""Given an ambiguous question, generate {self.num_interpretations} different specific SEARCH QUERIES (not answers) that clarify different meanings.

Original Question: {question}

For each interpretation, create a focused search query that:
1. Asks a specific question about one possible meaning
2. Is optimized for document retrieval (NOT an answer)
3. Does NOT include dates, names, or specific facts (those should be in retrieved documents)
4. Is concise (one question per line)

Example: If question is "When was X invented?", generate:
1. When was X first created?
2. What year did X get introduced?
3. When did X become available?

NOT: "X was invented in 1979" (that's an answer, not a query)

Output ONLY the search queries, one per line, numbered 1-{self.num_interpretations}. Do not include explanations, dates, or answers."""

        try:
            # Generate interpretations using LLM
            if hasattr(self.generator, 'client'):
                # OpenAI generator
                response = await self.generator.client.chat.completions.create(
                    model=self.generator.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that generates diverse interpretations of ambiguous questions. Respond with only the interpretations, one per line."
                        },
                        {
                            "role": "user",
                            "content": diversify_prompt
                        }
                    ],
                    max_tokens=200,
                    temperature=0.7  # Higher temperature for diversity
                )
                interpretations_text = response.choices[0].message.content.strip()
            else:
                # For local LLM, create a prompt that works with the generator
                local_prompt = f"""Given an ambiguous question, generate {self.num_interpretations} different specific SEARCH QUERIES (not answers) that clarify different meanings:

Original Question: {question}

For each interpretation, create a focused search query that:
1. Asks a specific question about one possible meaning
2. Is optimized for document retrieval (NOT an answer)
3. Does NOT include dates, names, or specific facts (those should be in retrieved documents)
4. Is concise (one question per line)

Example: If question is "When was X invented?", generate:
1. When was X first created?
2. What year did X get introduced?
3. When did X become available?

NOT: "X was invented in 1979" (that's an answer, not a query)

Output ONLY the search queries, one per line, numbered 1-{self.num_interpretations}."""
                interpretations_text = await self._generate_with_local_llm(
                    local_prompt,
                    max_tokens=200,
                    temperature=0.7  
                )

            # Parse interpretations
            interpretations = []
            for line in interpretations_text.split('\n'):
                line = line.strip()
                line = line.split('.', 1)[-1].strip() if '.' in line else line
                line = line.split('-', 1)[-1].strip() if '-' in line and line[0].isdigit() else line
                if line and len(line) > 10:  # Filter out empty or very short lines
                    interpretations.append(line)

            # Ensure we have at least one interpretation (fallback to original)
            if not interpretations:
                logger.warning("No interpretations generated, using original question")
                interpretations = [question]
            else:
                # Limit to num_interpretations
                interpretations = interpretations[:self.num_interpretations]

            logger.debug(f"Generated {len(interpretations)} interpretations: {interpretations}")
            return interpretations

        except Exception as e:
            logger.error(f"Error in diversification: {e}")
            # Fallback to original question
            return [question]

    async def _generate_with_local_llm(self, prompt: str, system_prompt: str = None, max_tokens: int = None, temperature: float = None) -> str:
        """
        Generate text using local LLM (LlamaCppGenerator).
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate (default: uses generator's max_tokens)
            temperature: Temperature override (default: uses generator's temperature)
            
        Returns:
            Generated text
        """
        try:
            # For local LLM, use the internal _generate_sync method directly
            loop = asyncio.get_event_loop()
            
            original_max_tokens = self.generator.max_tokens
            if max_tokens is not None:
                self.generator.max_tokens = max_tokens
            
            try:
                temp = temperature if temperature is not None else 0.0
                answer, _ = await loop.run_in_executor(
                    None,
                    self.generator._generate_sync,
                    prompt,
                    system_prompt,
                    temp,
                    None,  
                    None   
                )
                result = answer.strip()
            finally:
                if max_tokens is not None:
                    self.generator.max_tokens = original_max_tokens
            
            return result
        except Exception as e:
            logger.warning(f"Error generating with local LLM: {e}")
            # Fallback: return empty string so caller can handle fallback
            return ""

    async def _verify(
        self,
        question: str,
        retrieved_docs: List[RetrievalResult]
    ) -> Tuple[str, float]:
        """
        Stage 2: Verify - Assess quality of retrieved passages.
        
        Args:
            question: Original question
            retrieved_docs: Retrieved documents
            
        Returns:
            Tuple of (verification_label, confidence_score)
            - verification_label: "Useful", "Partially Useful", or "Useless"
            - confidence_score: Confidence in the verification (0.0 to 1.0)
        """
        if not retrieved_docs:
            return "Useless", 1.0

        # Prepare document summaries for verification
        doc_summaries = "\n\n".join([
            f"Document {i+1} (Title: {doc.title}, Score: {doc.score:.3f}):\n{doc.text[:200]}..."
            for i, doc in enumerate(retrieved_docs[:5])
        ])

        verify_prompt = f"""Assess whether the retrieved documents are useful for answering the question.

Question: {question}

Retrieved Documents:
{doc_summaries}

Classify the retrieval quality into one of these categories:
1. "Useful": Documents contain clear, relevant information to answer the question
2. "Partially Useful": Documents contain some relevant information but may be incomplete
3. "Useless": Documents do not contain relevant information to answer the question

Respond with ONLY the label (Useful, Partially Useful, or Useless) and a confidence score (0.0 to 1.0) separated by a comma.
Example: Useful, 0.85"""

        try:
            if hasattr(self.generator, 'client'):
                response = await self.generator.client.chat.completions.create(
                    model=self.generator.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that evaluates document relevance. Respond with only the label and confidence score separated by a comma."
                        },
                        {
                            "role": "user",
                            "content": verify_prompt
                        }
                    ],
                    max_tokens=50,
                    temperature=0.0  
                )
                verification_text = response.choices[0].message.content.strip()
            else:
                # Local LLM - use LLM to actually evaluate content relevance
                try:
                    system_prompt_verify = "You are a helpful assistant that evaluates document relevance. Respond with only the label and confidence score separated by a comma."
                    verification_text = await self._generate_with_local_llm(
                        verify_prompt, 
                        system_prompt=system_prompt_verify,
                        max_tokens=50,
                        temperature=0.0  # Deterministic for verification
                    )
                    # If empty response, fall through to fallback
                    if not verification_text or len(verification_text.strip()) < 5:
                        raise ValueError("Empty or invalid verification response")
                    logger.debug(f"Local LLM verification (content-based): {verification_text}")
                except Exception as e:
                    logger.warning(f"Local LLM verification failed ({e}), falling back to score-based heuristic")
                    # Fallback to score-based verification if LLM evaluation fails
                    avg_score = sum(doc.score for doc in retrieved_docs[:5]) / min(5, len(retrieved_docs)) if retrieved_docs else 0.0
                    
                    # Check document diversity (unique titles)
                    unique_titles = len(set(doc.title for doc in retrieved_docs[:5]))
                    diversity_ratio = unique_titles / min(5, len(retrieved_docs)) if retrieved_docs else 0.0
                    
                    # Normalize BM25 scores (they can be very high, e.g., 15+)
                    # Use a more conservative threshold that accounts for high BM25 scores
                    normalized_score = min(avg_score / 20.0, 1.0)  # Normalize to 0-1 range
                    combined_score = 0.5 * normalized_score + 0.5 * diversity_ratio
                    
                    # More conservative classification
                    if combined_score > 0.6:
                        verification_text = "Useful, 0.75"
                    elif combined_score > 0.3:
                        verification_text = "Partially Useful, 0.55"
                    else:
                        verification_text = "Useless, 0.65"
                    logger.debug(f"Local LLM verification (score-based fallback): {verification_text} (avg_score={avg_score:.3f}, diversity={diversity_ratio:.3f}, combined={combined_score:.3f})")

            # Parse verification result
            parts = verification_text.split(',')
            if len(parts) >= 2:
                label = parts[0].strip()
                try:
                    confidence = float(parts[1].strip())
                except ValueError:
                    confidence = 0.5
            else:
                # Try to extract label from text
                label = verification_text.strip()
                confidence = 0.5

            # Normalize label
            label_lower = label.lower()
            if "useful" in label_lower and "partially" not in label_lower:
                label = "Useful"
            elif "partially" in label_lower or "partial" in label_lower:
                label = "Partially Useful"
            else:
                label = "Useless"

            logger.debug(f"Verification: {label} (confidence: {confidence:.2f})")
            return label, confidence

        except Exception as e:
            logger.error(f"Error in verification: {e}")
            # Fallback: use average retrieval score
            if retrieved_docs:
                avg_score = sum(doc.score for doc in retrieved_docs[:5]) / min(5, len(retrieved_docs))
                if avg_score > 0.6:
                    return "Useful", 0.7
                elif avg_score > 0.3:
                    return "Partially Useful", 0.5
                else:
                    return "Useless", 0.6
            else:
                return "Useless", 1.0

    async def _adapt_generate(
        self,
        question: str,
        retrieved_docs: List[RetrievalResult],
        verification_label: str,
        dataset: str = None
    ) -> Tuple[str, float, int]:
        """
        Stage 3: Adapt - Generate answer using adaptive strategy.
        
        Args:
            question: Original question
            retrieved_docs: Retrieved documents
            verification_label: Verification label ("Useful", "Partially Useful", or "Useless")
            dataset: Dataset type ("ambignq" or "asqa"), defaults to self.dataset
            
        Returns:
            Tuple of (answer, generation_time, total_tokens)
        """
        if dataset is None:
            dataset = self.dataset
            
        if verification_label == "Useless":
            # Still use retrieval context, but note uncertainty
            # Even "poor" retrieval may contain useful information, and LLM can filter
            logger.debug("Adaptive strategy: Using RAG with uncertainty note (retrieval quality low)")
            
            # Use standard PromptRegistry prompts even for "Useless" classification
            try:
                answer, generation_time, total_tokens = await self.generator.generate(
                    question, retrieved_docs[:self.config.top_k] if retrieved_docs else [], dataset=dataset
                )
                return answer, generation_time, total_tokens
            except Exception as e:
                logger.error(f"Error in adaptive generation: {e}")
                # Fallback: use standard RAG
                answer, generation_time, total_tokens = await self.generator.generate(
                    question, retrieved_docs[:self.config.top_k] if retrieved_docs else [], dataset=dataset
                )
                return answer, generation_time, total_tokens
        else:
            # Use RAG (retrieved info + LLM knowledge)
            logger.debug(f"Adaptive strategy: Using RAG (verification: {verification_label})")
            
            # The PromptRegistry prompts are optimized for the dataset and perform best
            answer, generation_time, total_tokens = await self.generator.generate(
                question, retrieved_docs[:self.config.top_k], dataset=dataset
            )
            return answer, generation_time, total_tokens

    async def run_single(
        self,
        question: str,
        question_id: str,
        reference_data: Dict[str, Any]
    ) -> RAGResult:
        """
        Run DIVA RAG pipeline on a single question.
        
        Args:
            question: Question to answer
            question_id: Unique question identifier
            reference_data: Reference data for evaluation
            
        Returns:
            RAG result with answer and metrics
        """
        detected_type = detect_dataset_type(reference_data)
        dataset = detected_type.value
        
        total_retrieval_time = 0.0
        total_generation_time = 0.0
        total_tokens = 0

        # Stage 1: Diversify
        start_time = time.time()
        interpretations = await self._diversify(question)
        diversify_time = time.time() - start_time
        total_generation_time += diversify_time

        # Stage 2: Retrieve for original question AND interpretations, then merge
        # OPTIMIZATION: Include original question retrieval to match Vanilla quality
        # This ensures we don't lose the baseline quality while getting diversification benefits
        all_retrieved_docs = []
        seen_doc_ids = set()

        # Retrieve for original question first
        start_time = time.time()
        original_docs = self.retriever.retrieve(question)
        retrieval_time = time.time() - start_time
        total_retrieval_time += retrieval_time

        # Add original question's retrieval results first
        for doc in original_docs:
            all_retrieved_docs.append(doc)
            seen_doc_ids.add(doc.doc_id)

        # Retrieve for each interpretation 
        for interpretation in interpretations:
            start_time = time.time()
            retrieved_docs = self.retriever.retrieve(interpretation)
            retrieval_time = time.time() - start_time
            total_retrieval_time += retrieval_time

            # Merge: keep best score for each document (preserves ranking quality)
            for doc in retrieved_docs:
                if doc.doc_id not in seen_doc_ids:
                    # New document - add it
                    all_retrieved_docs.append(doc)
                    seen_doc_ids.add(doc.doc_id)
                else:
                    # Document already seen - keep the one with higher score
                    existing_idx = next(i for i, d in enumerate(all_retrieved_docs) if d.doc_id == doc.doc_id)
                    if doc.score > all_retrieved_docs[existing_idx].score:
                        all_retrieved_docs[existing_idx] = doc

        # Sort by original score (descending) and limit to top_k
        all_retrieved_docs.sort(key=lambda x: x.score, reverse=True)
        final_docs = all_retrieved_docs[:self.config.top_k]

        # Stage 3: Verify (optional)
        if self.skip_verification:
            # Skip verification - use default label and save API call
            verification_label = "Useful"
            verification_confidence = 1.0
            verify_time = 0.0
            logger.debug("Skipping verification step (--skip-verification enabled)")
        else:
            start_time = time.time()
            verification_label, verification_confidence = await self._verify(question, final_docs)
            verify_time = time.time() - start_time
            total_generation_time += verify_time

        # Stage 4: Adapt and Generate
        answer, gen_time, tokens = await self._adapt_generate(
            question, final_docs, verification_label, dataset=dataset
        )
        total_generation_time += gen_time
        total_tokens += tokens

        # Evaluation
        evaluation = self.evaluator.evaluate_single(
            prediction=answer,
            retrieved_docs=[doc.to_dict() for doc in final_docs],
            reference_item=reference_data,
            retrieval_time=total_retrieval_time,
            generation_time=total_generation_time,
            total_tokens=total_tokens
        )

        return RAGResult(
            question_id=question_id,
            question=question,
            retrieved_docs=[doc.to_dict() for doc in final_docs],
            generated_answer=answer,
            reference_data=reference_data,
            retrieval_time=total_retrieval_time,
            generation_time=total_generation_time,
            total_tokens=total_tokens,
            evaluation=evaluation,
            metadata={
                "interpretations": interpretations,
                "verification_label": verification_label,
                "verification_confidence": verification_confidence,
                "num_interpretations": len(interpretations),
                "total_docs_retrieved": len(all_retrieved_docs),
                "unique_docs_after_merge": len(final_docs),
                "diversify_time": diversify_time,
                "verify_time": verify_time,
                "retrieval_method": self.config.retrieval_mode,
                "score_normalization": False,
                "aggregation_method": "simple_merge_preserve_scores"
            }
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
        Run DIVA RAG pipeline on a batch of questions.
        
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
            desc=f"DIVA RAG ({self.config.retrieval_mode}, interpretations={self.num_interpretations})"
        )

        return results


def main():
    """Main entry point for DIVA RAG."""
    parser = argparse.ArgumentParser(
        description="DIVA (Diversify–Verify–Adapt) RAG with adaptive generation strategy"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="ambignq",
        choices=["ambignq", "asqa"],
        help="Dataset type (ambignq or asqa)"
    )

    # Data paths
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to test data (default: auto-select based on dataset)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save results (default: organized by approach/mode/model)"
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
        default="data/ambigqa_wiki.index",
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
        default="data/ambigqa_wiki_metadata.json",
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

    # DIVA-specific settings
    parser.add_argument(
        "--num-interpretations",
        type=int,
        default=3,
        help="Number of question interpretations to generate"
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip verification step (saves 1 API call per question, no performance impact)"
    )

    # Local LLM settings
    parser.add_argument(
        "--use-local-llm",
        action="store_true",
        help="Use local LLM via llama.cpp instead of OpenAI API"
    )
    parser.add_argument(
        "--local-model-path",
        type=str,
        default="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        help="Path to local GGUF model file"
    )
    parser.add_argument(
        "--local-context-size",
        type=int,
        default=4096,
        help="Context window size for local LLM"
    )
    parser.add_argument(
        "--local-gpu-layers",
        type=int,
        default=-1,
        help="Number of GPU layers to offload (-1 for all, uses Metal on M1)"
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--single-query",
        type=str,
        default=None,
        help="Run on a single query with full verbose output (exits after processing)"
    )

    args = parser.parse_args()

    # Setup logging level
    if args.verbose or args.single_query:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Load environment
    load_dotenv()

    # Handle single-query mode
    if args.single_query:
        # Single query test mode
        question = args.single_query
        logger.info(f"\n{'='*60}")
        logger.info(f"  DIVA RAG - Single Query Test")
        logger.info(f"{'='*60}\n")
        logger.info(f"Question: {question}\n")

        # Create config
        config = RAGConfig.from_args(args)
        # Use first mode if "all" is selected
        mode = args.retrieval_mode if args.retrieval_mode != "all" else "sparse"
        config.retrieval_mode = mode

        # Only require API key if not using local LLM
        if not args.use_local_llm:
            config.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not config.openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set. Use --use-local-llm to run without API.")

        # Initialize DIVA RAG
        rag = DIVARAG(
            config=config,
            num_interpretations=args.num_interpretations,
            use_local_llm=args.use_local_llm,
            local_model_path=args.local_model_path,
            local_context_size=args.local_context_size,
            local_gpu_layers=args.local_gpu_layers,
            skip_verification=getattr(args, 'skip_verification', False),
            dataset=getattr(args, 'dataset', 'ambignq')
        )

        # Load retriever
        logger.info(f"Loading retriever: {mode}")
        rag._load_retriever(mode)

        # Create dummy reference data for evaluation
        reference_data = {
            "question": question,
            "id": "single_query_test",
            "annotations": []  # Empty annotations for single query
        }

        # Run single query
        logger.info(f"\n{'='*60}")
        logger.info(f"  Running DIVA Pipeline")
        logger.info(f"{'='*60}\n")

        try:
            result = asyncio.run(rag.run_single(
                question=question,
                question_id="single_query_test",
                reference_data=reference_data
            ))

            # Print detailed results
            logger.info(f"\n{'='*60}")
            logger.info(f"  RESULTS")
            logger.info(f"{'='*60}\n")

            logger.info(f"Question: {result.question}\n")

            logger.info("Stage 1: Diversify")
            logger.info("-" * 60)
            interpretations = result.metadata.get("interpretations", [])
            for i, interpretation in enumerate(interpretations, 1):
                logger.info(f"  {i}. {interpretation}")
            logger.info(f"Diversify time: {result.metadata.get('diversify_time', 0):.2f}s\n")

            logger.info("Stage 2: Retrieve")
            logger.info("-" * 60)
            logger.info(f"Total documents retrieved: {result.metadata.get('total_docs_retrieved', 0)}")
            logger.info(f"Final documents used: {len(result.retrieved_docs)}")
            for i, doc in enumerate(result.retrieved_docs[:5], 1):
                logger.info(f"  {i}. {doc.get('title', 'Unknown')} (score: {doc.get('score', 0):.3f})")
            logger.info(f"Retrieval time: {result.retrieval_time:.2f}s\n")

            logger.info("Stage 3: Verify")
            logger.info("-" * 60)
            logger.info(f"Verification: {result.metadata.get('verification_label', 'Unknown')}")
            logger.info(f"Confidence: {result.metadata.get('verification_confidence', 0):.2f}")
            logger.info(f"Verify time: {result.metadata.get('verify_time', 0):.2f}s\n")

            logger.info("Stage 4: Adapt & Generate")
            logger.info("-" * 60)
            logger.info(f"Strategy: {result.metadata.get('verification_label', 'Unknown')}")
            logger.info(f"Generation time: {result.generation_time:.2f}s")
            logger.info(f"Total tokens: {result.total_tokens}\n")

            logger.info("Final Answer")
            logger.info("-" * 60)
            logger.info(f"{result.generated_answer}\n")

            logger.info("Performance Metrics")
            logger.info("-" * 60)
            logger.info(f"Total retrieval time: {result.retrieval_time:.2f}s")
            logger.info(f"Total generation time: {result.generation_time:.2f}s")
            logger.info(f"Total time: {result.retrieval_time + result.generation_time:.2f}s")
            logger.info(f"Total tokens: {result.total_tokens}\n")

            # Print evaluation if available
            if result.evaluation:
                logger.info("Evaluation Metrics")
                logger.info("-" * 60)
                for key, value in result.evaluation.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")

        except Exception as e:
            logger.error(f"Error processing single query: {e}", exc_info=True)
            return

        logger.info(f"\n{'='*60}\n")
        return  # Exit after single query

    # Set default data path based on dataset
    if args.data_path is None:
        if args.dataset == "asqa":
            args.data_path = "data/asqa_test.json"
        else:
            args.data_path = "data/ambignq_test.json"

    # Load test data (for batch mode)
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

        # Only require API key if not using local LLM
        if not args.use_local_llm:
            config.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not config.openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

        # Initialize DIVA RAG
        rag = DIVARAG(
            config=config,
            num_interpretations=args.num_interpretations,
            use_local_llm=args.use_local_llm,
            local_model_path=args.local_model_path,
            local_context_size=args.local_context_size,
            local_gpu_layers=args.local_gpu_layers,
            skip_verification=getattr(args, 'skip_verification', False),
            dataset=getattr(args, 'dataset', 'ambignq')
        )

        # Load retriever for this mode
        rag._load_retriever(mode)

        # Determine output path for resume capability
        if args.output_path is None:
            model_name = get_model_name_from_config(config.to_dict())
            is_test = args.limit is not None and args.limit < 100
            output_path = get_organized_output_path(
                approach="diva",
                retrieval_mode=mode,
                model_name=model_name,
                is_test=is_test,
                dataset=getattr(args, 'dataset', 'ambignq')
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
        config_dict = {**config.to_dict(), "num_interpretations": args.num_interpretations}
        merged_data = merge_results(existing_results, new_results, config_dict)

        # Recompute aggregate metrics on merged results
        logger.info(f"\nComputing aggregate metrics for {mode}...")
        aggregate_metrics = rag.evaluator.evaluate_batch(merged_data["results"])
        merged_data["aggregate_metrics"] = aggregate_metrics

        # Print report
        logger.info(f"\n{'='*60}\n  Results for {mode.upper()} mode\n{'='*60}")
        print_evaluation_report(aggregate_metrics)

        # Save results immediately
        ensure_output_directory(output_path)
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
        logger.info(f"\n{mode.upper()} results saved to {output_path}")

        # Save results
        all_results[mode] = merged_data

        # Cleanup if running multiple modes
        if args.retrieval_mode == "all":
            rag._cleanup_retriever()

        # Print cache stats
        if rag.cache:
            cache_stats = rag.cache.get_stats()
            logger.info(f"\nCache stats: {cache_stats}")

    # Final summary if running all modes
    if args.retrieval_mode == "all":
        logger.info(f"\n{'='*60}\n  ALL MODES COMPLETED\n{'='*60}")
        for mode in modes_to_run:
            data = all_results[mode]
            model_name = get_model_name_from_config(data["config"])
            is_test = args.limit is not None and args.limit < 100
            mode_output = get_organized_output_path(
                approach="diva",
                retrieval_mode=mode,
                model_name=model_name,
                is_test=is_test,
                dataset=getattr(args, 'dataset', 'ambignq')
            )
            logger.info(f"  - {mode_output}")


if __name__ == "__main__":
    main()

