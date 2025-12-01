"""
Prompt Registry for RAG Systems.

Centralizes all LLM prompts with dataset-specific variants for AmbigNQ and ASQA.
Provides a clean interface for retrieving prompts based on dataset type.
"""

from enum import Enum
from typing import List, Optional
from dataclasses import dataclass


class DatasetType(Enum):
    """Supported dataset types."""
    AMBIGNQ = "ambignq"
    ASQA = "asqa"


# =============================================================================
# AmbigNQ Prompts (Short, factual answers)
# =============================================================================

AMBIGNQ_GENERATION_SYSTEM = (
    "You are a helpful assistant that answers questions based on provided context."
)

AMBIGNQ_GENERATION_USER = """Using the context below, answer the question **only** with the final answer.

Context:
{context}

Question: {question}

Constraints:
- Output ONLY the entity (name, date, place, etc.).
- NO complete sentences (e.g., do not write "The answer is...").
- NO punctuation (except formatted dates/names).

Answer:"""

AMBIGNQ_HYDE_SYSTEM = "You are a helpful assistant that writes factual passages."

AMBIGNQ_HYDE_USER = """Write a brief, factual Wikipedia passage that explicitly answers this specific question.
Focus on including specific names, dates, or entities related to the query.

Question: {question}

Passage:"""

AMBIGNQ_SYNTHESIS_SYSTEM = None  # Instructions in user prompt

AMBIGNQ_SYNTHESIS_USER = """Question: {question}
Context: {context}

Analyze the context and extract the answer.
Output strictly in valid JSON format.

Structure your JSON exactly like this:
{{
  "intents": [
      {{"intent_label": "Short description of interpretation 1", "confidence": 1.0, "key_facts": ["fact1"]}}
  ],
  "synthesis": "A distinct explanation of the answer or ambiguity...",
  "concise_answer": "THE FINAL SHORT ANSWER STRING"
}}

JSON Output:"""

AMBIGNQ_SYNTHESIS_UNAMBIGUOUS_SUFFIX = """

The question appears unambiguous. Provide the direct factual answer."""

AMBIGNQ_SYNTHESIS_AMBIGUOUS_SUFFIX = """

The question is ambiguous. Consider these interpretations:
{subqueries}

Provide the best single answer in 'concise_answer', but explain the nuances in 'synthesis'."""

AMBIGNQ_SUBQUERY_FROM_CLUSTER = """Based on the following group of documents retrieved for the query "{question}", identify the specific interpretation they represent.

Documents:
{context}

Generate a SINGLE, specific question that focuses on this interpretation.
Output ONLY the question."""

AMBIGNQ_SUBQUERY_LLM_FALLBACK = """The following question is ambiguous. Generate 2-4 specific questions that represent different interpretations.

Original Question: {question}

Requirements:
- Each sub-query should be specific and unambiguous
- Cover the most likely interpretations

You MUST respond with ONLY valid JSON, no other text. Use this exact format:
{{
  "subqueries": ["specific question 1", "specific question 2", ...],
  "reasoning": "Brief explanation of the different interpretations"
}}"""

AMBIGNQ_QUERY_REFORMULATION_SYSTEM = (
    "You are a helpful assistant that reformulates search queries to find missing information. "
    "Respond with only the reformulated query."
)

AMBIGNQ_QUERY_REFORMULATION_USER = """You are a query reformulation expert. Given an original question and a partial answer, generate a more specific search query to find additional missing information.

Original Question: {question}

Current Answer: {previous_answer}

Analyze what information might be missing or incomplete in the current answer. Generate a focused search query that would help retrieve documents containing the missing information. The query should:
1. Be specific and targeted
2. Focus on gaps in the current answer
3. Be optimized for document retrieval
4. Be concise (1-2 sentences max)

Output ONLY the reformulated search query, nothing else."""


# =============================================================================
# ASQA Prompts (Long-form, comprehensive answers)
# =============================================================================

ASQA_GENERATION_SYSTEM = (
    "You are an expert Wikipedia editor. Your goal is to write comprehensive, "
    "neutral, and well-cited answers that resolve ambiguous questions."
)

ASQA_GENERATION_USER = """Using the provided context, write a single, cohesive paragraph (150-250 words) that answers the question by synthesizing all possible interpretations.

Context:
{context}

Question: {question}

Strict Requirements:
1. **Coverage**: You must address every distinct interpretation found in the context (e.g., "In the comic...", "In the film...").
2. **Structure**: Do NOT use bullet points or numbered lists. Write a flowing paragraph.
3. **Transitions**: Use contrastive transition words (e.g., "however", "conversely", "specifically", "while") to connect the different interpretations.
4. **No Fluff**: Start the answer directly. Do NOT say "Based on the documents" or "The context mentions."

Answer:"""

ASQA_HYDE_SYSTEM = "You are a helpful assistant that writes detailed, factual Wikipedia passages."

ASQA_HYDE_USER = """Write a high-quality Wikipedia passage that answers the question below. 
Since the question is likely ambiguous, the passage must explicitly mention and differentiate between multiple possible answers or interpretations (e.g., different people with the same name, different versions of a work).

Question: {question}

Passage (focus on distinguishing details):"""

ASQA_SYNTHESIS_SYSTEM = (
    "You are an expert question-answering system that provides comprehensive answers. "
    "Your answers should be detailed, well-organized, and address all interpretations of ambiguous questions."
)

ASQA_SYNTHESIS_USER = """Question: {question}
Context: {context}

Task:
Produce an ASQA-style synthesized answer.  
Your long answer must:
- Contain all facts needed for a downstream QA model to answer the question.
- Combine all interpretations into a single coherent passage.
- Be concise: **exactly 3–4 sentences**, no more.
- Use smooth transitions and avoid lists, bullets, or meta commentary.

Formatting rules:
- Output **only** valid JSON.
- No text before or after the JSON block.
- Keep "ambiguity_analysis" and "interpretations_found" short and precise.

JSON schema:
{{
  "ambiguity_analysis": "1–2 sentences explaining why the question is ambiguous.",
  "interpretations_found": [
    "Short label for Interpretation 1",
    "Short label for Interpretation 2"
  ],
  "long_answer": "A unified 3–4 sentence passage synthesizing all interpretations and containing all needed facts. No lists or filler."
}}

JSON Output:
"""

ASQA_SYNTHESIS_UNAMBIGUOUS_SUFFIX = """

The question appears to have a single interpretation. Provide a detailed, comprehensive answer with supporting evidence."""

ASQA_SYNTHESIS_AMBIGUOUS_SUFFIX = """

The question has multiple interpretations. Consider these specific aspects:
{subqueries}

Address each interpretation in your long_answer with specific evidence."""

ASQA_SUBQUERY_FROM_CLUSTER = """Based on the following group of documents retrieved for the query "{question}", identify the specific interpretation or aspect they represent.

Documents:
{context}

Generate a specific question that focuses on this interpretation/aspect. The question should help retrieve more detailed evidence.
Output ONLY the question."""

ASQA_SUBQUERY_LLM_FALLBACK = """The following question is ambiguous and may have multiple valid interpretations. Generate 2-4 specific questions that represent different interpretations or aspects.

Original Question: {question}

Requirements:
- Each sub-query should target a specific interpretation
- Cover the most important aspects that a comprehensive answer should address
- Sub-queries should help retrieve evidence for different facets of the answer

You MUST respond with ONLY valid JSON, no other text. Use this exact format:
{{
  "subqueries": ["specific question 1", "specific question 2", ...],
  "reasoning": "Brief explanation of the different interpretations/aspects"
}}"""

ASQA_QUERY_REFORMULATION_SYSTEM = (
    "You are a helpful assistant that reformulates search queries to find comprehensive information. "
    "Respond with only the reformulated query."
)

ASQA_QUERY_REFORMULATION_USER = """You are a query reformulation expert. Given an original question and a current answer draft, generate a search query to find additional information for a more comprehensive response.

Original Question: {question}

Current Answer Draft: {previous_answer}

Analyze what aspects or interpretations might be missing from the current answer. Generate a focused search query that would help retrieve documents to make the answer more comprehensive. The query should:
1. Target missing interpretations or aspects
2. Help find supporting evidence or additional facts
3. Be optimized for document retrieval
4. Be concise (1-2 sentences max)

Output ONLY the reformulated search query, nothing else."""


# =============================================================================
# Prompt Registry
# =============================================================================

@dataclass
class PromptSet:
    """Collection of prompts for a specific task."""
    system: Optional[str]
    user: str


class PromptRegistry:
    """
    Centralized registry for dataset-specific prompts.

    Usage:
        registry = PromptRegistry()
        prompt = registry.get_generation_prompt("asqa")
        formatted = prompt.user.format(context="...", question="...")
    """

    _PROMPTS = {
        DatasetType.AMBIGNQ: {
            "generation": PromptSet(
                system=AMBIGNQ_GENERATION_SYSTEM,
                user=AMBIGNQ_GENERATION_USER
            ),
            "hyde": PromptSet(
                system=AMBIGNQ_HYDE_SYSTEM,
                user=AMBIGNQ_HYDE_USER
            ),
            "synthesis": PromptSet(
                system=AMBIGNQ_SYNTHESIS_SYSTEM,
                user=AMBIGNQ_SYNTHESIS_USER
            ),
            "synthesis_unambiguous_suffix": AMBIGNQ_SYNTHESIS_UNAMBIGUOUS_SUFFIX,
            "synthesis_ambiguous_suffix": AMBIGNQ_SYNTHESIS_AMBIGUOUS_SUFFIX,
            "subquery_cluster": AMBIGNQ_SUBQUERY_FROM_CLUSTER,
            "subquery_fallback": AMBIGNQ_SUBQUERY_LLM_FALLBACK,
            "query_reformulation": PromptSet(
                system=AMBIGNQ_QUERY_REFORMULATION_SYSTEM,
                user=AMBIGNQ_QUERY_REFORMULATION_USER
            ),
        },
        DatasetType.ASQA: {
            "generation": PromptSet(
                system=ASQA_GENERATION_SYSTEM,
                user=ASQA_GENERATION_USER
            ),
            "hyde": PromptSet(
                system=ASQA_HYDE_SYSTEM,
                user=ASQA_HYDE_USER
            ),
            "synthesis": PromptSet(
                system=ASQA_SYNTHESIS_SYSTEM,
                user=ASQA_SYNTHESIS_USER
            ),
            "synthesis_unambiguous_suffix": ASQA_SYNTHESIS_UNAMBIGUOUS_SUFFIX,
            "synthesis_ambiguous_suffix": ASQA_SYNTHESIS_AMBIGUOUS_SUFFIX,
            "subquery_cluster": ASQA_SUBQUERY_FROM_CLUSTER,
            "subquery_fallback": ASQA_SUBQUERY_LLM_FALLBACK,
            "query_reformulation": PromptSet(
                system=ASQA_QUERY_REFORMULATION_SYSTEM,
                user=ASQA_QUERY_REFORMULATION_USER
            ),
        }
    }

    @classmethod
    def _normalize_dataset(cls, dataset: str) -> DatasetType:
        """Convert string to DatasetType enum."""
        if isinstance(dataset, DatasetType):
            return dataset
        dataset_lower = dataset.lower().strip()
        if dataset_lower in ("asqa",):
            return DatasetType.ASQA
        return DatasetType.AMBIGNQ  # Default to AmbigNQ

    @classmethod
    def get_generation_prompt(cls, dataset: str = "ambignq") -> PromptSet:
        """Get the generation prompt for a dataset."""
        dtype = cls._normalize_dataset(dataset)
        return cls._PROMPTS[dtype]["generation"]

    @classmethod
    def get_hyde_prompt(cls, dataset: str = "ambignq") -> PromptSet:
        """Get the HyDE prompt for a dataset."""
        dtype = cls._normalize_dataset(dataset)
        return cls._PROMPTS[dtype]["hyde"]

    @classmethod
    def get_synthesis_prompt(cls, dataset: str = "ambignq") -> PromptSet:
        """Get the synthesis prompt for a dataset."""
        dtype = cls._normalize_dataset(dataset)
        return cls._PROMPTS[dtype]["synthesis"]

    @classmethod
    def get_synthesis_suffix(cls, dataset: str = "ambignq", is_ambiguous: bool = True) -> str:
        """Get the synthesis suffix based on ambiguity detection."""
        dtype = cls._normalize_dataset(dataset)
        key = "synthesis_ambiguous_suffix" if is_ambiguous else "synthesis_unambiguous_suffix"
        return cls._PROMPTS[dtype][key]

    @classmethod
    def get_subquery_cluster_prompt(cls, dataset: str = "ambignq") -> str:
        """Get the subquery generation prompt for document clusters."""
        dtype = cls._normalize_dataset(dataset)
        return cls._PROMPTS[dtype]["subquery_cluster"]

    @classmethod
    def get_subquery_fallback_prompt(cls, dataset: str = "ambignq") -> str:
        """Get the LLM fallback prompt for subquery generation."""
        dtype = cls._normalize_dataset(dataset)
        return cls._PROMPTS[dtype]["subquery_fallback"]

    @classmethod
    def get_query_reformulation_prompt(cls, dataset: str = "ambignq") -> PromptSet:
        """Get the query reformulation prompt for iterative RAG."""
        dtype = cls._normalize_dataset(dataset)
        return cls._PROMPTS[dtype]["query_reformulation"]

    @classmethod
    def format_context(cls, contexts: List, max_chars: int = 500) -> str:
        """
        Format retrieved contexts into a string for prompts.

        Args:
            contexts: List of RetrievalResult objects or dicts
            max_chars: Maximum characters per document

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            if hasattr(ctx, 'title'):
                title = ctx.title
                text = ctx.text
            else:
                title = ctx.get('title', 'Unknown')
                text = ctx.get('text', '')

            text_truncated = text[:max_chars] if len(text) > max_chars else text
            context_parts.append(f"Document {i} (Title: {title}):\n{text_truncated}")

        return "\n\n".join(context_parts)


def detect_dataset_from_item(item: dict) -> DatasetType:
    """
    Auto-detect dataset type from a data item's structure.

    ASQA items have 'ambiguous_question' field or 'dataset': 'asqa'.
    AmbigNQ items have 'question' field with AmbigNQ-style annotations.

    Args:
        item: A single data item from the dataset

    Returns:
        DatasetType enum value
    """
    # Check explicit dataset marker (from our import script)
    if item.get("dataset") == "asqa":
        return DatasetType.ASQA

    # Check for ASQA-specific fields
    if "ambiguous_question" in item and "qa_pairs" in item:
        return DatasetType.ASQA

    # Check for ASQA-style annotations (long_answer field)
    annotations = item.get("annotations", [])
    if annotations and isinstance(annotations[0], dict):
        if "long_answer" in annotations[0]:
            return DatasetType.ASQA

    # Default to AmbigNQ
    return DatasetType.AMBIGNQ


def get_question_field(item: dict, dataset: Optional[str] = None) -> str:
    """
    Extract the question from a data item based on dataset type.

    Args:
        item: A single data item
        dataset: Optional dataset override

    Returns:
        The question string
    """
    if dataset:
        dtype = PromptRegistry._normalize_dataset(dataset)
    else:
        dtype = detect_dataset_from_item(item)

    if dtype == DatasetType.ASQA:
        return item.get("ambiguous_question") or item.get("question", "")
    return item.get("question", "")
