"""
Agentic Disambiguation - LangGraph Implementation

Multi-agent RAG pipeline for handling ambiguous questions using LangGraph:
1. Ambiguity detection: Determine if question has multiple interpretations (Aleatoric) or is Uncertain (Epistemic)
2. Sub-query decomposition: Break ambiguous question into specific sub-queries based on document clusters
3. HyDE generation: Generate hypothetical documents for each sub-query
4. Enhanced retrieval: Use HyDE docs to improve retrieval
5. Answer synthesis: Generate comprehensive answer covering all interpretations using Structured Output

This module uses LangGraph for agent orchestration and shared core components.
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
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Sequence, Literal, Tuple
import gc
import operator

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from pydantic import BaseModel, Field

from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

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
    ensure_output_directory,
    load_existing_results,
    get_processed_question_ids,
    filter_unprocessed_data,
    merge_results,
    detect_dataset_from_item,
    get_question_field,
    PromptRegistry,
)
from evaluation import RAGEvaluator, print_evaluation_report, compute_disambiguation_f1, compute_disambiguation_f1_asqa, detect_dataset_type, DatasetType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models for Structured Output
# ============================================================================

class IntentDetail(BaseModel):
    intent_label: str = Field(description="Label for this specific interpretation (e.g., 'Animal', 'Car')")
    confidence: float = Field(description="Confidence score for this interpretation")
    key_facts: List[str] = Field(description="List of key facts associated with this interpretation")

class MultiIntentResponse(BaseModel):
    intents: List[IntentDetail] = Field(description="List of identified intents/interpretations")
    synthesis: str = Field(description="The final amalgamated prose answer contrasting the interpretations")
    concise_answer: str = Field(description="A SINGLE concise, factual answer suitable for evaluation - just the facts, no narrative, no explanations. Must be ONE answer only.")
    disclaimer: Optional[str] = Field(description="Disclaimer if any intent was dropped or if information is missing")

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
    ambiguity_status: Literal["Unambiguous", "Ambiguous", "Uncertain"]
    is_ambiguous: bool # Kept for backward compatibility/reporting
    ambiguity_score: float
    ambiguity_reasoning: str

    # Sub-query generation
    subqueries: List[str]
    subquery_reasoning: str
    
    # Clusters (Intermediate state for hypothesis generation)
    clusters: Optional[List[Dict[str, Any]]]

    # HyDE generation
    hyde_documents: Dict[str, str]  # subquery -> hypothetical doc

    # Retrieval
    retrieved_docs: List[Dict[str, Any]]
    retrieval_time: float

    # Generation
    generated_answer: str # The concise answer for F1
    synthesis: str # The detailed synthesis for D-F1
    intents: List[Dict[str, Any]] # Structured intents
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
    1. Ambiguity Detection: Detect if question is ambiguous (Aleatoric) or Uncertain (Epistemic)
    2. Sub-query Generation: Decompose into specific sub-queries (with Relevance Pruning)
    3. HyDE Generation: Create hypothetical documents
    4. Enhanced Retrieval: Retrieve using sub-queries + HyDE
    5. Answer Synthesis: Generate comprehensive answer using Structured Output
    """

    def __init__(self, config: RAGConfig, dataset: str = "ambignq"):
        """Initialize LangGraph framework."""
        self.config = config
        self.dataset = dataset

        # Setup cache
        self.cache = RetrievalCache(
            cache_dir=config.cache_dir,
            enabled=config.use_cache
        ) if config.use_cache else None

        # Initialize retrievers (lazy loading)
        self.retriever = None
        
        # We need an encoder for the coherence check and relevance pruning.
        # We'll try to use the retriever's encoder if available, otherwise load one.
        self.encoder = None 
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder_model = SentenceTransformer(config.dense_encoder)
            logger.info(f"Encoder loaded: {config.dense_encoder}")
        except Exception as e:
            logger.warning(f"Could not load sentence-transformers: {e}. Some features may be limited.")
            self.encoder_model = None

        # Check if using local LLM
        if config.use_local_llm:
            if not LLAMA_CPP_AVAILABLE:
                raise RuntimeError(
                    "llama-cpp-python is not installed. "
                    "Install with: pip install llama-cpp-python"
                )

            logger.info("Initializing with LOCAL LLM (llama.cpp)")
            logger.info(f"Model path: {config.local_model_path}")

            # Initialize single local LLM generator with dataset awareness
            self.generator = LlamaCppGenerator(
                model_path=config.local_model_path,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                n_ctx=config.local_context_size,
                n_gpu_layers=config.local_gpu_layers,
                verbose=False,
                dataset=dataset  # Pass dataset for prompt selection
            )

            # HyDE generator uses same instance with dataset-aware prompts
            self.hyde_generator = self.generator
            self.llm = None

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
                api_key=config.openai_api_key,
                dataset=dataset
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
            
    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the loaded encoder."""
        if self.encoder_model:
            return self.encoder_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.array([])
        
    async def _generate_text(self, prompt: str, system_prompt: str = None) -> Tuple[str, int]:
        """
        Unified generation helper for both Local and API LLMs.
        
        Returns:
            Tuple of (generated_text, token_count)
        """
        if self.config.use_local_llm:
            loop = asyncio.get_event_loop()
            # LlamaCppGenerator._generate_sync returns (text, tokens)
            return await loop.run_in_executor(
                None, self.generator._generate_sync, prompt, system_prompt
            )
        else:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            response = await self.llm.ainvoke(messages)
            token_count = 0
            if hasattr(response, 'response_metadata'):
                token_count = response.response_metadata.get('token_usage', {}).get('total_tokens', 0)
            return response.content.strip(), token_count

    def _build_synthesis_prompt(self, question: str, subqueries: List[str], docs_context: str) -> str:
        """Build prompt for answer synthesis using PromptRegistry."""
        is_unambiguous = len(subqueries) == 1 and subqueries[0] == question

        # Get dataset-specific synthesis prompt
        prompt_set = PromptRegistry.get_synthesis_prompt(self.dataset)
        base_prompt = prompt_set.user.format(question=question, context=docs_context)

        # Add appropriate suffix based on ambiguity
        suffix = PromptRegistry.get_synthesis_suffix(self.dataset, is_ambiguous=not is_unambiguous)
        if not is_unambiguous:
            subq_list = "\n".join(f"- {sq}" for sq in subqueries)
            suffix = suffix.format(subqueries=subq_list)

        return base_prompt + suffix

    # ------------------------------------------------------------------------
    # Helper Functions for Ambiguity & Pruning
    # ------------------------------------------------------------------------

    def _assess_retrieval_validity(self, docs: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Assess the validity of retrieved documents using Coherence Check.
        
        Uses dataset-aware thresholds:
        - AmbigNQ: Standard thresholds (questions may or may not be ambiguous)
        - ASQA: Lower thresholds (all questions are designed to be ambiguous)
        """
        if not docs or not self.encoder_model:
            return {"status": "Uncertain", "reason": "No docs or no encoder"}

        doc_texts = [d.text for d in docs]
        embeddings = self._encode_text(doc_texts)
        
        if len(embeddings) < 2:
             return {"status": "Unambiguous", "reason": "Too few docs to check coherence"}

        # 1. Calculate Centroid
        centroid = np.mean(embeddings, axis=0)
        
        # 2. Calculate Variance (avg euclidean distance to centroid)
        distances = euclidean_distances(embeddings, [centroid])
        avg_distance = np.mean(distances)
        
        # 3. Cluster Separability
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(embeddings)
            if len(set(kmeans.labels_)) > 1:
                separability = silhouette_score(embeddings, kmeans.labels_)
            else:
                separability = 0.0
        except Exception:
            separability = 0.0

        # Dataset-aware thresholds
        # ASQA: All questions are inherently ambiguous, use much lower thresholds
        # AmbigNQ: Use standard thresholds
        if self.dataset == "asqa":
            # ASQA: Assume ambiguity by default since all questions are designed to be ambiguous
            # Only mark as unambiguous if separability is very low AND variance is low
            VARIANCE_THRESHOLD = 0.5  # Lower threshold for ASQA
            SEPARABILITY_THRESHOLD = 0.0  # Effectively always detect ambiguity for ASQA
            
            # For ASQA, default to Ambiguous unless very strong evidence otherwise
            status = "Ambiguous"
            reason = f"ASQA default ambiguous (Variance: {avg_distance:.2f}, Separability: {separability:.2f})"
            
            if avg_distance < 0.3 and separability < 0.0:
                # Very homogeneous docs - might be unambiguous
                status = "Unambiguous"
                reason = f"Low variance ({avg_distance:.2f}) suggests single interpretation"
            elif avg_distance > 1.0 and separability < 0.0:
                # High variance but no clear clusters - epistemic uncertainty
                status = "Uncertain"
                reason = f"Epistemic Failure: High Variance ({avg_distance:.2f}) & Low Separability ({separability:.2f})"
        else:
            # AmbigNQ: Standard thresholds
            VARIANCE_THRESHOLD = 0.8 
            SEPARABILITY_THRESHOLD = 0.1 
            
            status = "Unambiguous"
            reason = f"Variance: {avg_distance:.2f}, Separability: {separability:.2f}"

            if avg_distance > VARIANCE_THRESHOLD:
                if separability < SEPARABILITY_THRESHOLD:
                    status = "Uncertain"
                    reason = f"Epistemic Failure: High Variance ({avg_distance:.2f}) & Low Separability ({separability:.2f})"
                else:
                    status = "Ambiguous"
                    reason = f"Ambiguous: Distinct clusters detected (Sep: {separability:.2f})"
            elif separability > SEPARABILITY_THRESHOLD:
                 status = "Ambiguous"
                 reason = f"Ambiguous: Distinct clusters detected (Sep: {separability:.2f})"
        
        return {
            "status": status,
            "variance": avg_distance,
            "separability": separability,
            "centroid": centroid,
            "reason": reason
        }

    # ------------------------------------------------------------------------
    # LangGraph Node Functions
    # ------------------------------------------------------------------------

    async def detect_ambiguity_node(self, state: AgentState) -> AgentState:
        """
        Node 1: Detect Ambiguity with Coherence Check.
        """
        logger.debug(f"\n{'='*60}\n[STEP 1] AMBIGUITY DETECTION & COHERENCE CHECK\n{'='*60}")
        question = state["question"]
        logger.debug(f"  Question: \"{question}\"")
        
        # 1. Initial Retrieval
        start_time = time.time()
        retrieved_docs = self.retriever.retrieve(question, k=self.config.top_k * 2)
        retrieval_time = time.time() - start_time
        
        # 2. Coherence Check
        validity = self._assess_retrieval_validity(retrieved_docs)
        status = validity["status"]

        logger.debug(f"  Retrieved {len(retrieved_docs)} initial docs in {retrieval_time:.3f}s")
        logger.debug(f"  Coherence Check Results:")
        logger.debug(f"    - Status: {status}")
        logger.debug(f"    - Variance: {validity.get('variance', 'N/A'):.3f}" if isinstance(validity.get('variance'), (int, float)) else f"    - Variance: {validity.get('variance', 'N/A')}")
        logger.debug(f"    - Separability: {validity.get('separability', 'N/A'):.3f}" if isinstance(validity.get('separability'), (int, float)) else f"    - Separability: {validity.get('separability', 'N/A')}")
        logger.debug(f"    - Reason: {validity['reason']}")

        if status == "Uncertain":
            logger.warning(f"Epistemic Failure detected! {validity['reason']}")

        is_ambiguous = status in ("Ambiguous", "Uncertain")
        
        return {
            **state,
            "is_ambiguous": is_ambiguous,
            "ambiguity_status": status,
            "ambiguity_score": validity.get("separability", 0.0),
            "ambiguity_reasoning": validity["reason"],
            "retrieved_docs": [doc.to_dict() for doc in retrieved_docs[:self.config.top_k]],
            "retrieval_time": retrieval_time,
            "messages": [HumanMessage(content=f"Ambiguity Status: {status}. {validity['reason']}")],
        }

    async def generate_subqueries_node(self, state: AgentState) -> AgentState:
        """
        Node 2: Generate Hypotheses (Sub-queries) with Relevance Pruning.
        """
        question = state["question"]
        retrieved_docs = [RetrievalResult(**d) for d in state["retrieved_docs"]]
        
        if not retrieved_docs or not self.encoder_model:
            logger.warning("Skipping cluster-based generation due to missing docs or encoder.")
            return await self._generate_subqueries_llm_fallback(state)

        logger.debug(f"\n{'='*60}\n[STEP 2] HYPOTHESIS GENERATION (CLUSTERING & PRUNING)\n{'='*60}")

        # 1. Cluster Documents
        doc_texts = [d.text for d in retrieved_docs]
        doc_embeddings = self._encode_text(doc_texts)
        query_embedding = self._encode_text([question])[0]
        
        # Determine K
        k = min(max(2, len(doc_texts) // 2), 4)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(doc_embeddings)

        logger.debug(f"  Clustering {len(doc_texts)} documents into {k} clusters")

        # 2. Relevance Pruning
        RELEVANCE_THRESHOLD = 0.2
        cluster_relevances = []
        for i in range(k):
            relevance = cosine_similarity([query_embedding], [kmeans.cluster_centers_[i]])[0][0]
            cluster_relevances.append((i, relevance))
            cluster_size = sum(1 for label in kmeans.labels_ if label == i)
            status = "✓ KEPT" if relevance >= RELEVANCE_THRESHOLD else "✗ PRUNED"
            logger.debug(f"    - Cluster {i}: relevance={relevance:.3f}, size={cluster_size} docs [{status}]")

        valid_clusters = [i for i, rel in cluster_relevances if rel >= RELEVANCE_THRESHOLD]

        logger.debug(f"  Pruning Result: {len(valid_clusters)}/{k} clusters kept (threshold={RELEVANCE_THRESHOLD})")

        if not valid_clusters:
            logger.warning("All clusters pruned! Falling back to original question.")
            return {
                **state,
                "subqueries": [question],
                "subquery_reasoning": "All clusters pruned due to low relevance.",
                "messages": [HumanMessage(content="Relevance Pruning: All clusters rejected.")],
            }

        # 3. Generate Sub-queries for Valid Clusters
        logger.debug(f"  Generating sub-queries for {len(valid_clusters)} valid clusters...")
        generated_subqueries = []
        cluster_descriptions = []

        for cluster_idx in valid_clusters:
            cluster_docs = [retrieved_docs[j] for j, label in enumerate(kmeans.labels_) if label == cluster_idx]
            context_str = "\n".join([f"- {d.title}: {d.text[:200]}..." for d in cluster_docs[:3]])

            # Use dataset-specific subquery cluster prompt
            prompt_template = PromptRegistry.get_subquery_cluster_prompt(self.dataset)
            prompt = prompt_template.format(question=question, context=context_str)

            subq, _ = await self._generate_text(prompt)
            generated_subqueries.append(subq)
            cluster_descriptions.append(f"Cluster {cluster_idx}: {subq}")
            logger.debug(f"    - Cluster {cluster_idx} → \"{subq[:80]}...\"" if len(subq) > 80 else f"    - Cluster {cluster_idx} → \"{subq}\"")

        logger.debug(f"  Generated {len(generated_subqueries)} sub-queries total")

        return {
            **state,
            "subqueries": generated_subqueries,
            "subquery_reasoning": "\n".join(cluster_descriptions),
            "messages": [HumanMessage(content=f"Generated {len(generated_subqueries)} sub-queries from clusters")],
        }

    async def _generate_subqueries_llm_fallback(self, state: AgentState) -> AgentState:
        """Fallback to original LLM-based generation if clustering fails."""
        question = state["question"]
        # Use dataset-specific fallback prompt
        prompt_template = PromptRegistry.get_subquery_fallback_prompt(self.dataset)
        prompt = prompt_template.format(question=question)

        try:
            response_text, _ = await self._generate_text(prompt)

            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:].strip()

            result = json.loads(response_text)
            subqueries = result.get("subqueries", [question])
            return {
                **state,
                "subqueries": subqueries,
                "subquery_reasoning": result.get("reasoning", "Fallback generation"),
                "messages": [HumanMessage(content=f"Generated {len(subqueries)} sub-queries (fallback)")],
            }
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return {**state, "subqueries": [question], "error": str(e)}

    async def generate_hyde_docs_node(self, state: AgentState) -> AgentState:
        """
        Node 3: Generate HyDE documents for each sub-query.
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
            hyde_documents = {sq: sq for sq in subqueries}
            return {
                **state,
                "hyde_documents": hyde_documents,
                "error": str(e),
            }

    async def retrieve_with_hyde_node(self, state: AgentState) -> AgentState:
        """
        Node 4: Enhanced retrieval using sub-queries and HyDE documents.
        """
        subqueries = state["subqueries"]
        hyde_documents = state["hyde_documents"]

        start_time = time.time()
        all_retrieved_docs = []

        try:
            for subquery in subqueries:
                hyde_doc = hyde_documents.get(subquery, subquery)
                subquery_results = self.retriever.retrieve(subquery, k=self.config.top_k)
                hyde_results = self.retriever.retrieve(hyde_doc, k=self.config.top_k)
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
        """Node 5: Synthesize answer using Structured Output."""
        question = state["question"]
        subqueries = state["subqueries"]
        retrieved_docs = state["retrieved_docs"]

        docs_context = "\n\n".join([
            f"Document {i+1} (Title: {d.get('title', '')}):\n{d['text'][:500]}"
            for i, d in enumerate(retrieved_docs)
        ])

        try:
            start_time = time.time()
            prompt_text = self._build_synthesis_prompt(question, subqueries, docs_context)
            
            response_text, tokens = await self._generate_text(prompt_text)
            
            # Parse JSON output
            try:
                # Clean markdown if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                data = json.loads(response_text.strip())

                # Handle dataset-specific JSON structures
                if self.dataset == "asqa":
                    # ASQA format: {"ambiguity_analysis", "interpretations_found", "short_answers_extracted", "long_answer"}
                    long_answer = data.get("long_answer", "")
                    concise_answer = long_answer  # For ASQA, the long answer IS the answer
                    synthesis = long_answer
                    # Extract short answers for debugging and potential use in evaluation
                    short_answers_extracted = data.get("short_answers_extracted", [])
                    # Convert interpretations to intents format for consistency
                    interpretations = data.get("interpretations_found", [])
                    intents = [{"intent_label": interp, "confidence": 1.0, "key_facts": short_answers_extracted} for interp in interpretations]
                    ambiguity_analysis = data.get("ambiguity_analysis", "")
                    logger.debug(f"ASQA Ambiguity Analysis: {ambiguity_analysis[:100]}...")
                    logger.debug(f"ASQA Short Answers Extracted: {short_answers_extracted}")
                else:
                    # AmbigNQ format: {"intents", "synthesis", "concise_answer"}
                    concise_answer = data.get("concise_answer", "")
                    synthesis = data.get("synthesis", "")
                    intents = data.get("intents", [])
            except json.JSONDecodeError:
                # Fallback simple extraction
                logger.warning(f"Failed to parse JSON from response: {response_text[:100]}...")
                concise_answer = response_text[:100] if self.dataset != "asqa" else response_text
                synthesis = response_text
                intents = []

            logger.debug(f"Final answer: '{concise_answer[:100]}...'" if len(concise_answer) > 100 else f"Final answer: '{concise_answer}'")

            return {
                **state,
                "generated_answer": concise_answer,
                "synthesis": synthesis,
                "intents": intents,
                "generation_time": time.time() - start_time,
                "total_tokens": tokens,
                "messages": [HumanMessage(content="Generated answer via Structured Output")],
            }
        except Exception as e:
            logger.error(f"Error in answer synthesis: {e}")
            return {
                **state, 
                "generated_answer": f"Error: {str(e)}", 
                "synthesis": "", 
                "intents": [], 
                "generation_time": 0.0, 
                "total_tokens": 0, 
                "error": str(e)
            }

    def should_decompose(self, state: AgentState) -> str:
        """
        Conditional edge: Decide whether to decompose question.
        
        For ASQA: Always decompose since all questions are designed to be ambiguous.
        For AmbigNQ: Use ambiguity detection results.
        """
        status = state.get("ambiguity_status", "Ambiguous")
        
        # ASQA: Always decompose - all questions are inherently ambiguous
        # Only fallback for severe epistemic failure
        if self.dataset == "asqa":
            if status == "Uncertain":
                # Even for epistemic uncertainty, try decomposition for ASQA
                logger.info("ASQA: Forcing decomposition despite epistemic uncertainty")
                return "generate_subqueries"
            return "generate_subqueries"
        
        # AmbigNQ: Use standard ambiguity detection
        if status == "Uncertain":
            return "simple_retrieval"  # Fallback for epistemic failure
        return "generate_subqueries" if state.get("is_ambiguous", True) else "simple_retrieval"

    async def simple_retrieval_node(self, state: AgentState) -> AgentState:
        """
        Alternative path: Simple retrieval for unambiguous questions.
        
        For ASQA: Enhanced retrieval using HyDE even in simple path.
        For AmbigNQ: Standard retrieval.
        """
        question = state["question"]
        existing_docs = state.get("retrieved_docs", [])
        
        if existing_docs:
            return {
                **state,
                "subqueries": [question],
                "hyde_documents": {question: question},
                "messages": [HumanMessage(content=f"Using {len(existing_docs)} cached documents")],
            }

        start_time = time.time()
        try:
            # For ASQA: Use HyDE-enhanced retrieval even in simple path
            if self.dataset == "asqa":
                # Generate HyDE document for better retrieval
                hyde_doc, _, _ = await self.hyde_generator.generate_hypothetical_document(question)
                
                # Retrieve using both original question and HyDE doc
                query_results = self.retriever.retrieve(question, k=self.config.top_k)
                hyde_results = self.retriever.retrieve(hyde_doc, k=self.config.top_k)
                
                # Deduplicate and merge
                doc_map = {}
                for doc in query_results + hyde_results:
                    if doc.doc_id not in doc_map or doc.score > doc_map[doc.doc_id].score:
                        doc_map[doc.doc_id] = doc
                
                results = sorted(doc_map.values(), key=lambda x: x.score, reverse=True)[:self.config.top_k]
                hyde_documents = {question: hyde_doc}
                logger.debug(f"ASQA simple path: HyDE-enhanced retrieval, {len(results)} docs")
            else:
                # AmbigNQ: Standard retrieval
                results = self.retriever.retrieve(question, k=self.config.top_k)
                hyde_documents = {question: question}
            
            return {
                **state,
                "subqueries": [question],
                "hyde_documents": hyde_documents,
                "retrieved_docs": [doc.to_dict() for doc in results],
                "retrieval_time": time.time() - start_time,
                "messages": [HumanMessage(content=f"Simple retrieval: {len(results)} documents")],
            }
        except Exception as e:
            logger.error(f"Error in simple retrieval: {e}")
            return {**state, "retrieved_docs": [], "retrieval_time": time.time() - start_time, "error": str(e)}

    # ------------------------------------------------------------------------
    # Workflow Construction
    # ------------------------------------------------------------------------

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow.
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
        Run the LangGraph workflow on a single question (retrieval + generation only).

        Evaluation is decoupled and runs in batch after all generations complete.
        This improves performance when using heavy evaluation models like RoBERTa.
        """
        # Initialize state
        initial_state: AgentState = {
            "question": question,
            "question_id": question_id,
            "reference_data": reference_data,
            "ambiguity_status": "Unambiguous", # Default
            "is_ambiguous": False,
            "ambiguity_score": 0.0,
            "ambiguity_reasoning": "",
            "subqueries": [],
            "subquery_reasoning": "",
            "clusters": None,
            "hyde_documents": {},
            "retrieved_docs": [],
            "retrieval_time": 0.0,
            "generated_answer": "",
            "synthesis": "",
            "intents": [],
            "generation_time": 0.0,
            "total_tokens": 0,
            "evaluation": {},
            "messages": [],
            "error": None,
        }

        try:
            # Run workflow
            final_state = await self.workflow.ainvoke(initial_state)

            generated_answer = final_state["generated_answer"]
            synthesis = final_state["synthesis"]

            # Note: Evaluation is decoupled - runs in batch after all generations complete
            return RAGResult(
                question_id=question_id,
                question=question,
                retrieved_docs=final_state["retrieved_docs"],
                generated_answer=generated_answer,
                reference_data=reference_data,
                retrieval_time=final_state["retrieval_time"],
                generation_time=final_state["generation_time"],
                total_tokens=final_state["total_tokens"],
                evaluation={},  # Filled in post-processing batch evaluation
                metadata={
                    "ambiguity_status": final_state.get("ambiguity_status"),
                    "is_ambiguous": final_state["is_ambiguous"],
                    "ambiguity_score": final_state["ambiguity_score"],
                    "ambiguity_reasoning": final_state["ambiguity_reasoning"],
                    "num_subqueries": len(final_state["subqueries"]),
                    "subqueries": final_state["subqueries"],
                    "subquery_reasoning": final_state["subquery_reasoning"],
                    "synthesis": synthesis,
                    "intents": final_state["intents"],
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

    async def _process_with_semaphore(self, item: Dict[str, Any], semaphore: asyncio.Semaphore) -> RAGResult:
        """Process a single item with concurrency control."""
        question = get_question_field(item, self.dataset)
        question_id = item.get('id', str(hash(question)))
        async with semaphore:
            try:
                return await self.run_single(question, question_id, item)
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                return RAGResult(
                    question_id=question_id, question=question, retrieved_docs=[],
                    generated_answer=f"ERROR: {str(e)}", reference_data=item,
                    retrieval_time=0.0, generation_time=0.0, total_tokens=0,
                    evaluation={}, metadata={"error": str(e)}
                )

    async def run_batch(self, test_data: List[Dict[str, Any]], limit: Optional[int] = None) -> List[RAGResult]:
        """Run LangGraph workflow on a batch of questions."""
        if limit:
            test_data = test_data[:limit]
        semaphore = asyncio.Semaphore(self.config.concurrency)
        tasks = [self._process_with_semaphore(item, semaphore) for item in test_data]
        return await async_tqdm.gather(*tasks, desc=f"LangGraph Agentic ({self.config.retrieval_mode})")


def main():
    """Main entry point for agentic disambiguation."""
    parser = argparse.ArgumentParser(
        description="Agentic Disambiguation Framework (Novel Approach)"
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="ambignq", choices=["ambignq", "asqa"], help="Dataset type")

    # Data paths
    parser.add_argument("--data-path", type=str, default=None, help="Path to test data (default: auto-select based on dataset)")
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

    # Set default data path based on dataset
    if args.data_path is None:
        if args.dataset == "asqa":
            args.data_path = "data/asqa_test.json"
        else:
            args.data_path = "data/ambignq_test.json"

    # Load test data
    logger.info(f"Loading {args.dataset.upper()} test data from {args.data_path}...")
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
        framework = LangGraphAgenticDisambiguation(config, dataset=args.dataset)
        framework._load_retriever(mode)

        # Determine output path for resume capability
        if args.output_path is None:
            model_name = get_model_name_from_config(config.to_dict())
            is_test = args.limit is not None and args.limit < 100
            output_path = get_organized_output_path(
                approach="agentic",
                retrieval_mode=mode,
                model_name=model_name,
                is_test=is_test,
                dataset=args.dataset
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
            results = asyncio.run(framework.run_batch(test_data, limit=args.limit))
            new_results = [r.to_dict() for r in results]

            # Post-generation batch evaluation (decoupled for performance)
            logger.info(f"\nRunning post-generation evaluation for {mode}...")
            new_results = framework.evaluator.evaluate_results_post_generation(
                new_results, show_progress=True
            )

            # Add synthesis D-F1 for agentic results (secondary evaluation)
            logger.info(f"Computing synthesis D-F1 for agentic results...")
            for result in new_results:
                synthesis = result.get("metadata", {}).get("synthesis", "")
                if synthesis and result.get("evaluation"):
                    reference_data = result.get("reference_data", {})
                    # Use dataset-appropriate D-F1 computation
                    dataset_type = detect_dataset_type(reference_data)
                    if dataset_type == DatasetType.ASQA:
                        # ASQA uses qa_pairs with QA-based extraction
                        d_f1, covered, total = compute_disambiguation_f1_asqa(
                            synthesis,
                            reference_data,
                            threshold=config.d_f1_threshold,
                            use_qa_model=True
                        )
                    else:
                        # AmbigNQ uses annotations with token overlap
                        annotations = reference_data.get("annotations", [])
                        d_f1, covered, total = compute_disambiguation_f1(
                            synthesis,
                            annotations,
                            threshold=config.d_f1_threshold
                        )
                    result["evaluation"]["synthesis_d_f1"] = d_f1
                    result["evaluation"]["synthesis_covered"] = covered
                    result["evaluation"]["synthesis_total"] = total
        else:
            logger.info(f"All items already processed, skipping run")
            new_results = []

        # Merge with existing results
        config_dict = config.to_dict()
        merged_data = merge_results(existing_results, new_results, config_dict)
        merged_data["implementation"] = "LangGraph"

        # Recompute aggregate metrics on merged results
        logger.info(f"\nComputing aggregate metrics for {mode}...")
        aggregate_metrics = framework.evaluator.evaluate_batch(merged_data["results"])
        
        # Calculate Epistemic Uncertainty
        uncertain_count = sum(1 for r in merged_data["results"] if r.get("metadata", {}).get("ambiguity_status") == "Uncertain")
        aggregate_metrics["epistemic_uncertainty_count"] = uncertain_count
        aggregate_metrics["epistemic_uncertainty_rate"] = uncertain_count / len(merged_data["results"]) if merged_data["results"] else 0.0

        merged_data["aggregate_metrics"] = aggregate_metrics

        # Print report
        logger.info(f"\n{'='*60}\n  Results for {mode.upper()} mode (LangGraph)\n{'='*60}")
        print_evaluation_report(aggregate_metrics)
        logger.info(f"Epistemic Uncertainty: {uncertain_count}/{len(merged_data['results'])} ({aggregate_metrics['epistemic_uncertainty_rate']:.2%})")

        # Save results
        all_results[mode] = merged_data

        # Save results immediately
        ensure_output_directory(output_path)
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
        logger.info(f"\n{mode.upper()} results saved to {output_path}")

        # Cleanup
        if args.retrieval_mode == "all":
            framework._cleanup_retriever()

        # Cache stats
        if framework.cache:
            cache_stats = framework.cache.get_stats()
            logger.info(f"\nCache stats: {cache_stats}")

    # Results already saved during loop, just log summary
        if args.retrieval_mode == "all":
            logger.info(f"\n{'='*60}\n  ALL MODES COMPLETED\n{'='*60}")
            if args.output_path is None:
                data = all_results[modes_to_run[0]]
                model_name = get_model_name_from_config(data["config"])
                is_test = args.limit is not None and args.limit < 100
                base_output = get_organized_output_path(
                    approach="agentic",
                    retrieval_mode=modes_to_run[0],
                    model_name=model_name,
                    is_test=is_test,
                    dataset=args.dataset
                )
                logger.info(f"\nAll results saved in: {base_output.parent}")
                for mode in modes_to_run:
                    mode_output = get_organized_output_path(
                        approach="agentic",
                        retrieval_mode=mode,
                        model_name=model_name,
                        is_test=is_test,
                        dataset=args.dataset
                    )
                    logger.info(f"  - {mode_output.name}")
    else:
        # Results already saved, just log
        if args.output_path is None:
            data = all_results[args.retrieval_mode]
            model_name = get_model_name_from_config(data["config"])
            is_test = args.limit is not None and args.limit < 100
            output_path = get_organized_output_path(
                approach="agentic",
                retrieval_mode=args.retrieval_mode,
                model_name=model_name,
                is_test=is_test,
                dataset=args.dataset
            )
        else:
            output_path = Path(args.output_path)
        
        logger.info(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
