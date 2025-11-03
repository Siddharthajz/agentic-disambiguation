"""
Agentic Disambiguation - LangGraph Implementation

Multi-agent RAG pipeline for handling ambiguous questions using LangGraph:
1. Ambiguity detection: Determine if question has multiple interpretations
2. Sub-query decomposition: Break ambiguous question into specific sub-queries
3. HyDE generation: Generate hypothetical documents for each sub-query
4. Enhanced retrieval: Use HyDE docs to improve retrieval
5. Answer synthesis: Generate comprehensive answer covering all interpretations

This module uses LangGraph for agent orchestration and shared core components.
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Sequence
import gc
import operator

from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from core import (
    RAGConfig,
    RetrievalCache,
    create_retriever,
    OpenAIGenerator,
    HyDEGenerator,
    LlamaCppGenerator,
    LLAMA_CPP_AVAILABLE,
    RAGResult,
    RetrievalResult,
    get_model_name_from_config,
    get_organized_output_path,
    ensure_output_directory
)
from evaluation import RAGEvaluator, print_evaluation_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# LangGraph State Schema
# ============================================================================

class AgentState(TypedDict):
    """State for the LangGraph workflow."""
    # Input
    question: str
    question_id: str
    reference_data: Dict[str, Any]

    # Ambiguity detection
    is_ambiguous: bool
    ambiguity_score: float
    ambiguity_reasoning: str

    # Sub-query generation
    subqueries: List[str]
    subquery_reasoning: str

    # HyDE generation
    hyde_documents: Dict[str, str]  # subquery -> hypothetical doc

    # Retrieval
    retrieved_docs: List[Dict[str, Any]]
    retrieval_time: float

    # Generation
    generated_answer: str
    generation_time: float
    total_tokens: int

    # Evaluation
    evaluation: Dict[str, Any]

    # Messages for LangChain compatibility
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Error tracking
    error: Optional[str]


# ============================================================================
# LangGraph-Based Implementation
# ============================================================================

class LangGraphAgenticDisambiguation:
    """
    LangGraph-based agentic disambiguation framework.

    Uses LangGraph's StateGraph to orchestrate a multi-agent workflow:
    1. Ambiguity Detection: Detect if question is ambiguous
    2. Sub-query Generation: Decompose into specific sub-queries
    3. HyDE Generation: Create hypothetical documents
    4. Enhanced Retrieval: Retrieve using sub-queries + HyDE
    5. Answer Synthesis: Generate comprehensive answer
    """

    def __init__(self, config: RAGConfig):
        """Initialize LangGraph framework."""
        self.config = config

        # Setup cache
        self.cache = RetrievalCache(
            cache_dir=config.cache_dir,
            enabled=config.use_cache
        ) if config.use_cache else None

        # Initialize retrievers (lazy loading)
        self.retriever = None

        # Check if using local LLM
        if config.use_local_llm:
            if not LLAMA_CPP_AVAILABLE:
                raise RuntimeError(
                    "llama-cpp-python is not installed. "
                    "Install with: pip install llama-cpp-python"
                )

            logger.info("Initializing with LOCAL LLM (llama.cpp)")
            logger.info(f"Model path: {config.local_model_path}")

            # Initialize single local LLM generator (shared for all tasks)
            # Using a single instance avoids memory issues with llama.cpp
            self.generator = LlamaCppGenerator(
                model_path=config.local_model_path,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                n_ctx=config.local_context_size,
                n_gpu_layers=config.local_gpu_layers,
                verbose=False
            )

            # Reuse the same generator for HyDE (just change temperature at call time)
            # This avoids loading the model twice
            self.hyde_generator = self.generator

            # For sub-query generation, we'll use the local LLM directly in the node
            self.llm = None  # Will handle manually in generate_subqueries_node

        else:
            logger.info("Initializing with OpenAI API")

            # Initialize LangChain LLM
            self.llm = ChatOpenAI(
                model=config.llm_model,
                temperature=config.temperature,
                api_key=config.openai_api_key
            )

            # Initialize OpenAI generators
            self.generator = OpenAIGenerator(
                model=config.llm_model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                api_key=config.openai_api_key
            )

            self.hyde_generator = HyDEGenerator(
                model=config.llm_model,
                max_tokens=config.max_tokens,
                temperature=0.7,
                api_key=config.openai_api_key
            )

        # Initialize evaluator
        self.evaluator = RAGEvaluator(
            k=config.top_k,
            d_f1_threshold=config.d_f1_threshold
        )

        # Build LangGraph workflow
        self.workflow = self._build_workflow()

        logger.info("LangGraph Agentic Disambiguation initialized")

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

    # ------------------------------------------------------------------------
    # LangGraph Node Functions
    # ------------------------------------------------------------------------

    async def detect_ambiguity_node(self, state: AgentState) -> AgentState:
        """
        Node 1: Detect if the question is ambiguous.

        TODO: Implement LLM-based ambiguity detection.
        For now, assumes all questions are ambiguous to ensure comprehensive
        handling of potential ambiguities in the AmbigNQ dataset.

        Future implementation could use LLM to analyze:
        - Underspecified entities (e.g., "When did the US enter the war?" - which war?)
        - Multiple valid time frames, locations, or contexts
        - Different possible interpretations
        """
        logger.debug(f"\n{'='*60}\n[STEP 1] AMBIGUITY DETECTION\n{'='*60}")
        logger.debug(f"Question: {state['question']}")

        # Placeholder: Assume all questions are ambiguous
        # This ensures the full agentic pipeline runs for all examples
        is_ambiguous = True
        ambiguity_score = 1.0
        reasoning = "Placeholder: All questions treated as potentially ambiguous"

        logger.debug(f"Result: {'AMBIGUOUS' if is_ambiguous else 'NOT AMBIGUOUS'}")
        logger.debug(f"Confidence: {ambiguity_score:.2f}")
        logger.debug(f"Reasoning: {reasoning}")

        return {
            **state,
            "is_ambiguous": is_ambiguous,
            "ambiguity_score": ambiguity_score,
            "ambiguity_reasoning": reasoning,
            "messages": [HumanMessage(content=f"Ambiguity detection: {'ambiguous' if is_ambiguous else 'not ambiguous'} (placeholder)")],
        }

    async def generate_subqueries_node(self, state: AgentState) -> AgentState:
        """
        Node 2: Generate sub-queries for different interpretations.

        Decomposes the ambiguous question into 2-4 specific sub-queries
        representing different interpretations.
        """
        question = state["question"]

        prompt = f"""The following question is ambiguous. Generate 2-4 specific questions that represent different interpretations.

Original Question: {question}

Requirements:
- Each sub-query should be specific and unambiguous
- Cover the most likely interpretations
- Make explicit what was implicit in the original

You MUST respond with ONLY valid JSON, no other text. Use this exact format:
{{
  "subqueries": ["specific question 1", "specific question 2", ...],
  "reasoning": "Brief explanation of the different interpretations"
}}"""

        try:
            # Handle local LLM vs OpenAI
            if self.config.use_local_llm:
                # Use local LLM directly (synchronous, but wrapped in executor)
                loop = asyncio.get_event_loop()
                response_text, _ = await loop.run_in_executor(
                    None,
                    self.generator._generate_sync,
                    prompt,
                    None  # No system prompt
                )
            else:
                # Use LangChain OpenAI
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                response_text = response.content.strip()

            # Try to extract JSON if wrapped in markdown code blocks
            if response_text.startswith("```"):
                # Remove markdown code blocks
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:].strip()

            result = json.loads(response_text)

            subqueries = result.get("subqueries", [question])
            if not subqueries:
                subqueries = [question]

            return {
                **state,
                "subqueries": subqueries,
                "subquery_reasoning": result.get("reasoning", ""),
                "messages": [HumanMessage(content=f"Generated {len(subqueries)} sub-queries")],
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in sub-query generation: {e}")
            logger.debug(f"Response content: {response.content[:200]}")
            return {
                **state,
                "subqueries": [question],  # Fallback to original
                "subquery_reasoning": f"JSON parsing error: {str(e)}",
                "error": str(e),
            }
        except Exception as e:
            logger.error(f"Error in sub-query generation: {e}")
            return {
                **state,
                "subqueries": [question],  # Fallback to original
                "subquery_reasoning": f"Error: {str(e)}",
                "error": str(e),
            }

    async def generate_hyde_docs_node(self, state: AgentState) -> AgentState:
        """
        Node 3: Generate HyDE documents for each sub-query.

        Creates hypothetical documents that would answer each sub-query.
        """
        subqueries = state["subqueries"]
        hyde_documents = {}

        try:
            for subquery in subqueries:
                doc, gen_time, tokens = await self.hyde_generator.generate_hypothetical_document(subquery)
                hyde_documents[subquery] = doc
                logger.debug(f"Generated HyDE doc for: {subquery[:50]}...")

            return {
                **state,
                "hyde_documents": hyde_documents,
                "messages": [HumanMessage(content=f"Generated {len(hyde_documents)} HyDE documents")],
            }
        except Exception as e:
            logger.error(f"Error in HyDE generation: {e}")
            # Fallback: use sub-queries as "documents"
            hyde_documents = {sq: sq for sq in subqueries}
            return {
                **state,
                "hyde_documents": hyde_documents,
                "error": str(e),
            }

    async def retrieve_with_hyde_node(self, state: AgentState) -> AgentState:
        """
        Node 4: Enhanced retrieval using sub-queries and HyDE documents.

        Retrieves documents using both sub-queries and their hypothetical
        documents, then merges and deduplicates results.
        """
        subqueries = state["subqueries"]
        hyde_documents = state["hyde_documents"]

        start_time = time.time()
        all_retrieved_docs = []

        try:
            for subquery in subqueries:
                hyde_doc = hyde_documents.get(subquery, subquery)

                # Retrieve using sub-query
                subquery_results = self.retriever.retrieve(subquery, k=self.config.top_k)

                # Retrieve using HyDE document
                hyde_results = self.retriever.retrieve(hyde_doc, k=self.config.top_k)

                # Merge results
                all_retrieved_docs.extend(subquery_results)
                all_retrieved_docs.extend(hyde_results)

            # Deduplicate by doc_id and re-rank by score
            doc_map = {}
            for doc in all_retrieved_docs:
                if doc.doc_id not in doc_map or doc.score > doc_map[doc.doc_id].score:
                    doc_map[doc.doc_id] = doc

            unique_docs = sorted(doc_map.values(), key=lambda x: x.score, reverse=True)[:self.config.top_k]
            retrieval_time = time.time() - start_time

            return {
                **state,
                "retrieved_docs": [doc.to_dict() for doc in unique_docs],
                "retrieval_time": retrieval_time,
                "messages": [HumanMessage(content=f"Retrieved {len(unique_docs)} unique documents")],
            }
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return {
                **state,
                "retrieved_docs": [],
                "retrieval_time": time.time() - start_time,
                "error": str(e),
            }

    async def synthesize_answer_node(self, state: AgentState) -> AgentState:
        """
        Node 5: Synthesize comprehensive answer covering all interpretations.

        Generates an answer that addresses all sub-queries using the
        retrieved documents.
        """
        question = state["question"]
        subqueries = state["subqueries"]
        retrieved_docs = state["retrieved_docs"]

        # Convert back to RetrievalResult objects
        docs = [
            RetrievalResult(
                doc_id=d["doc_id"],
                title=d.get("title", ""),
                text=d["text"],
                score=d["score"],
                rank=d.get("rank", i),
                source=d.get("source", "hybrid")
            )
            for i, d in enumerate(retrieved_docs)
        ]

        try:
            start_time = time.time()

            # Enhanced prompt for multi-interpretation answers
            enhanced_question = f"""{question}

This question has multiple interpretations:
{chr(10).join(f"- {sq}" for sq in subqueries)}

Please provide a comprehensive answer that addresses all interpretations."""

            answer, gen_time, tokens = await self.generator.generate(enhanced_question, docs)
            generation_time = time.time() - start_time

            return {
                **state,
                "generated_answer": answer,
                "generation_time": generation_time,
                "total_tokens": tokens,
                "messages": [HumanMessage(content="Generated comprehensive answer")],
            }
        except Exception as e:
            logger.error(f"Error in answer synthesis: {e}")
            return {
                **state,
                "generated_answer": f"Error generating answer: {str(e)}",
                "generation_time": 0.0,
                "total_tokens": 0,
                "error": str(e),
            }

    def should_decompose(self, state: AgentState) -> str:
        """
        Conditional edge: Decide whether to decompose question.

        If ambiguous, proceed to sub-query generation.
        If not ambiguous, skip to direct retrieval.
        """
        if state.get("is_ambiguous", True):
            return "generate_subqueries"
        else:
            return "simple_retrieval"

    async def simple_retrieval_node(self, state: AgentState) -> AgentState:
        """
        Alternative path: Simple retrieval for unambiguous questions.

        Skip sub-query generation and HyDE, just retrieve directly.
        """
        question = state["question"]

        start_time = time.time()
        try:
            results = self.retriever.retrieve(question, k=self.config.top_k)
            retrieval_time = time.time() - start_time

            return {
                **state,
                "subqueries": [question],
                "hyde_documents": {question: question},
                "retrieved_docs": [doc.to_dict() for doc in results],
                "retrieval_time": retrieval_time,
                "messages": [HumanMessage(content=f"Simple retrieval: {len(results)} documents")],
            }
        except Exception as e:
            logger.error(f"Error in simple retrieval: {e}")
            return {
                **state,
                "retrieved_docs": [],
                "retrieval_time": time.time() - start_time,
                "error": str(e),
            }

    # ------------------------------------------------------------------------
    # Workflow Construction
    # ------------------------------------------------------------------------

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Workflow:
        START -> detect_ambiguity -> [conditional]
                                    -> if ambiguous: generate_subqueries -> generate_hyde -> retrieve_hyde -> synthesize
                                    -> if not: simple_retrieval -> synthesize
                                    -> END
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("detect_ambiguity", self.detect_ambiguity_node)
        workflow.add_node("generate_subqueries", self.generate_subqueries_node)
        workflow.add_node("generate_hyde", self.generate_hyde_docs_node)
        workflow.add_node("retrieve_hyde", self.retrieve_with_hyde_node)
        workflow.add_node("simple_retrieval", self.simple_retrieval_node)
        workflow.add_node("synthesize_answer", self.synthesize_answer_node)

        # Set entry point
        workflow.set_entry_point("detect_ambiguity")

        # Add conditional edge after ambiguity detection
        workflow.add_conditional_edges(
            "detect_ambiguity",
            self.should_decompose,
            {
                "generate_subqueries": "generate_subqueries",
                "simple_retrieval": "simple_retrieval",
            }
        )

        # Add edges for ambiguous path
        workflow.add_edge("generate_subqueries", "generate_hyde")
        workflow.add_edge("generate_hyde", "retrieve_hyde")
        workflow.add_edge("retrieve_hyde", "synthesize_answer")

        # Add edge for simple path
        workflow.add_edge("simple_retrieval", "synthesize_answer")

        # End after synthesis
        workflow.add_edge("synthesize_answer", END)

        return workflow.compile()

    # ------------------------------------------------------------------------
    # Main Pipeline Methods
    # ------------------------------------------------------------------------

    async def run_single(
        self,
        question: str,
        question_id: str,
        reference_data: Dict[str, Any]
    ) -> RAGResult:
        """
        Run the LangGraph workflow on a single question.

        Args:
            question: Question to answer
            question_id: Unique question identifier
            reference_data: Reference data for evaluation

        Returns:
            RAG result with comprehensive answer
        """
        # Initialize state
        initial_state: AgentState = {
            "question": question,
            "question_id": question_id,
            "reference_data": reference_data,
            "is_ambiguous": False,
            "ambiguity_score": 0.0,
            "ambiguity_reasoning": "",
            "subqueries": [],
            "subquery_reasoning": "",
            "hyde_documents": {},
            "retrieved_docs": [],
            "retrieval_time": 0.0,
            "generated_answer": "",
            "generation_time": 0.0,
            "total_tokens": 0,
            "evaluation": {},
            "messages": [],
            "error": None,
        }

        try:
            # Run workflow
            final_state = await self.workflow.ainvoke(initial_state)

            # Evaluate
            evaluation = self.evaluator.evaluate_single(
                prediction=final_state["generated_answer"],
                retrieved_docs=final_state["retrieved_docs"],
                reference_item=reference_data,
                retrieval_time=final_state["retrieval_time"],
                generation_time=final_state["generation_time"],
                total_tokens=final_state["total_tokens"]
            )

            return RAGResult(
                question_id=question_id,
                question=question,
                retrieved_docs=final_state["retrieved_docs"],
                generated_answer=final_state["generated_answer"],
                reference_data=reference_data,
                retrieval_time=final_state["retrieval_time"],
                generation_time=final_state["generation_time"],
                total_tokens=final_state["total_tokens"],
                evaluation=evaluation,
                metadata={
                    "is_ambiguous": final_state["is_ambiguous"],
                    "ambiguity_score": final_state["ambiguity_score"],
                    "ambiguity_reasoning": final_state["ambiguity_reasoning"],
                    "num_subqueries": len(final_state["subqueries"]),
                    "subqueries": final_state["subqueries"],
                    "subquery_reasoning": final_state["subquery_reasoning"],
                    "error": final_state.get("error"),
                }
            )
        except Exception as e:
            logger.error(f"Error in workflow execution: {e}")
            return RAGResult(
                question_id=question_id,
                question=question,
                retrieved_docs=[],
                generated_answer=f"ERROR: {str(e)}",
                reference_data=reference_data,
                retrieval_time=0.0,
                generation_time=0.0,
                total_tokens=0,
                evaluation={},
                metadata={"error": str(e)}
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
            return await self.run_single(question, question_id, item)

    async def run_batch(
        self,
        test_data: List[Dict[str, Any]],
        limit: Optional[int] = None
    ) -> List[RAGResult]:
        """
        Run LangGraph workflow on a batch of questions.

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
            desc=f"LangGraph Agentic ({self.config.retrieval_mode})"
        )

        return results




def main():
    """Main entry point for agentic disambiguation."""
    parser = argparse.ArgumentParser(
        description="Agentic Disambiguation Framework (Novel Approach)"
    )

    # Data paths
    parser.add_argument("--data-path", type=str, default="data/ambignq_test.json")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save results (default: organized by approach/mode/model)")

    # Retrieval settings
    parser.add_argument("--retrieval-mode", type=str, default="hybrid", choices=["sparse", "dense", "hybrid", "all"])
    parser.add_argument("--sparse-index", type=str, default="wikipedia-dpr")
    parser.add_argument("--dense-index", type=str, default="data/ambigqa_wiki.index", help="Path to FAISS index file")
    parser.add_argument("--dense-encoder", type=str, default="all-MiniLM-L6-v2", help="Sentence-transformers model name for query encoding")
    parser.add_argument("--dense-metadata", type=str, default="data/ambigqa_wiki_metadata.json", help="Path to metadata JSON file for dense retrieval")
    parser.add_argument("--top-k", type=int, default=5)

    # Generation settings
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-tokens", type=int, default=200)

    # Local LLM settings (optional - no API required)
    parser.add_argument("--use-local-llm", action="store_true", help="Use local LLM via llama.cpp instead of OpenAI API")
    parser.add_argument("--local-model-path", type=str, default="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf", help="Path to GGUF model file")
    parser.add_argument("--local-context-size", type=int, default=8192, help="Context window size for local LLM")
    parser.add_argument("--local-gpu-layers", type=int, default=-1, help="GPU layers to offload (-1 for all, uses Metal on M1)")

    # Performance settings
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--no-cache", action="store_false", dest="use_cache")

    # Experiment settings
    parser.add_argument("--limit", type=int, default=None)

    # Debug and logging settings
    parser.add_argument("--verbose", action="store_true", help="Enable detailed debug logging for each step")
    parser.add_argument("--single-query", type=str, default=None, help="Run on a single query with full verbose output")

    args = parser.parse_args()

    # Enable debug logging if verbose mode
    if args.verbose or args.single_query:
        logger.setLevel(logging.DEBUG)

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

    logger.info("\nUsing LangGraph implementation\n")

    for mode in modes_to_run:
        logger.info(f"\n{'='*60}\n  Running {mode.upper()} retrieval mode (LangGraph)\n{'='*60}\n")

        # Create config
        config = RAGConfig.from_args(args)
        config.retrieval_mode = mode

        # Only require OpenAI API key if not using local LLM
        if not args.use_local_llm:
            config.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not config.openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set. Use --use-local-llm to run without API.")

        # Initialize LangGraph framework
        framework = LangGraphAgenticDisambiguation(config)
        framework._load_retriever(mode)

        # Run experiments
        results = asyncio.run(framework.run_batch(test_data, limit=args.limit))

        # Compute metrics
        logger.info(f"\nComputing aggregate metrics for {mode}...")
        aggregate_metrics = framework.evaluator.evaluate_batch([r.to_dict() for r in results])

        # Print report
        logger.info(f"\n{'='*60}\n  Results for {mode.upper()} mode (LangGraph)\n{'='*60}")
        print_evaluation_report(aggregate_metrics)

        # Save results
        all_results[mode] = {
            "config": config.to_dict(),
            "aggregate_metrics": aggregate_metrics,
            "results": [r.to_dict() for r in results],
            "implementation": "LangGraph"
        }

        # Cleanup
        if args.retrieval_mode == "all":
            framework._cleanup_retriever()

        # Cache stats
        if framework.cache:
            cache_stats = framework.cache.get_stats()
            logger.info(f"\nCache stats: {cache_stats}")

    # Save results
    if args.retrieval_mode == "all":
        for mode, data in all_results.items():
            # Use organized path if custom output path not specified
            if args.output_path is None:
                model_name = get_model_name_from_config(data["config"])
                is_test = args.limit is not None and args.limit < 100
                mode_output = get_organized_output_path(
                    approach="agentic",
                    retrieval_mode=mode,
                    model_name=model_name,
                    is_test=is_test
                )
                ensure_output_directory(mode_output)
            else:
                output_path = Path(args.output_path)
                mode_output = output_path.parent / f"{output_path.stem}_{mode}{output_path.suffix}"
                output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(mode_output, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"\n{mode.upper()} results saved to {mode_output}")
    else:
        # Use organized path if custom output path not specified
        if args.output_path is None:
            data = all_results[args.retrieval_mode]
            model_name = get_model_name_from_config(data["config"])
            is_test = args.limit is not None and args.limit < 100
            output_path = get_organized_output_path(
                approach="agentic",
                retrieval_mode=args.retrieval_mode,
                model_name=model_name,
                is_test=is_test
            )
            ensure_output_directory(output_path)
        else:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(all_results[args.retrieval_mode], f, indent=2)
        logger.info(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
