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
from typing import List, Dict, Any, Optional, Tuple, TypedDict, Annotated, Sequence
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

        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            api_key=config.openai_api_key
        )

        # Initialize generators (for HyDE and final answer)
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

        Uses LLM to analyze the question and determine if it has multiple
        valid interpretations.
        """
        question = state["question"]

        prompt = f"""Analyze the following question and determine if it is ambiguous (has multiple valid interpretations).

Question: {question}

Consider:
- Does the question contain underspecified entities (e.g., "When did the US enter the war?" - which war?)
- Are there multiple valid time frames, locations, or contexts?
- Could different people interpret this question differently?

Respond in JSON format:
{{
  "is_ambiguous": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why the question is or isn't ambiguous"
}}"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)

            return {
                **state,
                "is_ambiguous": result["is_ambiguous"],
                "ambiguity_score": result["confidence"],
                "ambiguity_reasoning": result["reasoning"],
                "messages": [HumanMessage(content=f"Ambiguity detected: {result['is_ambiguous']}")],
            }
        except Exception as e:
            logger.error(f"Error in ambiguity detection: {e}")
            return {
                **state,
                "is_ambiguous": True,  # Default to ambiguous
                "ambiguity_score": 0.5,
                "ambiguity_reasoning": f"Error in detection: {str(e)}",
                "error": str(e),
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

Respond in JSON format:
{{
  "subqueries": ["specific question 1", "specific question 2", ...],
  "reasoning": "Brief explanation of the different interpretations"
}}"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)

            subqueries = result.get("subqueries", [question])
            if not subqueries:
                subqueries = [question]

            return {
                **state,
                "subqueries": subqueries,
                "subquery_reasoning": result.get("reasoning", ""),
                "messages": [HumanMessage(content=f"Generated {len(subqueries)} sub-queries")],
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
                text=d["text"],
                score=d["score"],
                metadata=d.get("metadata", {})
            )
            for d in retrieved_docs
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


# ============================================================================
# Original Implementation (for comparison)
# ============================================================================

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

    # Implementation selection
    parser.add_argument(
        "--use-langgraph",
        action="store_true",
        default=False,
        help="Use LangGraph-based implementation (default: use original implementation)"
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

    # Select implementation
    implementation_name = "LangGraph" if args.use_langgraph else "Original"
    logger.info(f"\nUsing {implementation_name} implementation\n")

    for mode in modes_to_run:
        logger.info(f"\n{'='*60}\n  Running {mode.upper()} retrieval mode ({implementation_name})\n{'='*60}\n")

        # Create config
        config = RAGConfig.from_args(args)
        config.retrieval_mode = mode
        config.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Initialize agentic framework (choose implementation)
        if args.use_langgraph:
            framework = LangGraphAgenticDisambiguation(config)
        else:
            framework = AgenticDisambiguation(config)

        framework._load_retriever(mode)

        # Run experiments
        results = asyncio.run(framework.run_batch(test_data, limit=args.limit))

        # Compute metrics
        logger.info(f"\nComputing aggregate metrics for {mode}...")
        aggregate_metrics = framework.evaluator.evaluate_batch([r.to_dict() for r in results])

        # Print report
        logger.info(f"\n{'='*60}\n  Results for {mode.upper()} mode ({implementation_name})\n{'='*60}")
        print_evaluation_report(aggregate_metrics)

        # Save results
        all_results[mode] = {
            "config": config.to_dict(),
            "aggregate_metrics": aggregate_metrics,
            "results": [r.to_dict() for r in results],
            "implementation": implementation_name
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
    if args.use_langgraph:
        logger.info("Completed LangGraph-based agentic disambiguation")
        logger.info("Workflow includes:")
        logger.info("  1. Ambiguity Detection (LLM-based)")
        logger.info("  2. Sub-query Generation (LLM-based)")
        logger.info("  3. HyDE Document Generation")
        logger.info("  4. Enhanced Retrieval (sub-queries + HyDE)")
        logger.info("  5. Answer Synthesis")
    else:
        logger.info("NOTE: Original implementation is a skeleton.")
        logger.info("Use --use-langgraph for full LangGraph orchestration")
    logger.info("="*60)


if __name__ == "__main__":
    main()
