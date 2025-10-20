"""
Generator components for RAG systems.

Provides modular generator implementations:
- BaseGenerator: Abstract base class
- OpenAIGenerator: OpenAI API implementation with async batching
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import openai
import tiktoken

from .data_models import RetrievalResult

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Abstract base class for answer generators."""

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 200, temperature: float = 0.0):
        """
        Initialize generator.

        Args:
            model: Model name/identifier
            max_tokens: Maximum tokens in generation
            temperature: Sampling temperature
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    async def generate(
        self,
        question: str,
        contexts: List[RetrievalResult]
    ) -> Tuple[str, float, int]:
        """
        Generate answer for a question given retrieved contexts.

        Args:
            question: Question to answer
            contexts: Retrieved documents as context

        Returns:
            Tuple of (answer, generation_time, total_tokens)
        """
        pass

    def format_prompt(self, question: str, contexts: List[RetrievalResult]) -> str:
        """
        Format prompt for generation.

        Args:
            question: Question to answer
            contexts: Retrieved contexts

        Returns:
            Formatted prompt string
        """
        context_str = "\n\n".join([
            f"Document {ctx.rank} (Title: {ctx.title}):\n{ctx.text[:500]}"
            for ctx in contexts
        ])

        prompt = f"""Using the context below, answer the question **only** with the final answer.
- Be concise and factual.
- Do not include explanations, reasoning, or extra text.
- Respond in a single line suitable for automatic comparison with ground truth.

Context:
{context_str}

Question: {question}

Answer:"""
        return prompt


class OpenAIGenerator(BaseGenerator):
    """OpenAI API generator with async support."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        max_tokens: int = 200,
        temperature: float = 0.0,
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI generator.

        Args:
            model: OpenAI model name
            max_tokens: Max tokens for generation
            temperature: Sampling temperature
            api_key: OpenAI API key (uses env var if None)
        """
        super().__init__(model=model, max_tokens=max_tokens, temperature=temperature)

        self.client = openai.AsyncOpenAI(api_key=api_key)

        # Load tokenizer for counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(f"OpenAI generator initialized: {model}")

    async def generate(
        self,
        question: str,
        contexts: List[RetrievalResult]
    ) -> Tuple[str, float, int]:
        """
        Generate answer using OpenAI API.

        Args:
            question: Question to answer
            contexts: Retrieved contexts

        Returns:
            Tuple of (answer, generation_time, total_tokens)
        """
        prompt = self.format_prompt(question, contexts)

        start_time = time.time()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            answer = response.choices[0].message.content.strip()
            total_tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            answer = f"ERROR: {str(e)}"
            total_tokens = 0

        generation_time = time.time() - start_time
        return answer, generation_time, total_tokens

    async def generate_batch(
        self,
        questions: List[str],
        contexts_list: List[List[RetrievalResult]],
        concurrency: int = 10
    ) -> List[Tuple[str, float, int]]:
        """
        Generate answers for multiple questions with concurrency control.

        Args:
            questions: List of questions
            contexts_list: List of context lists (one per question)
            concurrency: Maximum concurrent requests

        Returns:
            List of (answer, generation_time, total_tokens) tuples
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def generate_with_semaphore(q, ctxs):
            async with semaphore:
                return await self.generate(q, ctxs)

        tasks = [
            generate_with_semaphore(q, ctxs)
            for q, ctxs in zip(questions, contexts_list)
        ]

        results = await asyncio.gather(*tasks)
        return results


class HyDEGenerator(OpenAIGenerator):
    """
    HyDE (Hypothetical Document Embeddings) generator.

    Generates hypothetical documents that would answer the question,
    which can then be used for improved retrieval.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        max_tokens: int = 200,
        temperature: float = 0.7,  # Higher temp for diversity
        api_key: Optional[str] = None
    ):
        """
        Initialize HyDE generator.

        Args:
            model: OpenAI model name
            max_tokens: Max tokens for generation
            temperature: Sampling temperature (higher for diversity)
            api_key: OpenAI API key
        """
        super().__init__(model=model, max_tokens=max_tokens, temperature=temperature, api_key=api_key)
        logger.info("HyDE generator initialized")

    async def generate_hypothetical_document(self, question: str) -> Tuple[str, float, int]:
        """
        Generate a hypothetical document that would answer the question.

        Args:
            question: Question to generate document for

        Returns:
            Tuple of (hypothetical_document, generation_time, total_tokens)
        """
        prompt = f"""Write a brief, factual passage that would appear in Wikipedia and contain the answer to this question:

Question: {question}

Write a 2-3 sentence passage that directly answers this question:"""

        start_time = time.time()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that writes factual passages."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            document = response.choices[0].message.content.strip()
            total_tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0

        except Exception as e:
            logger.error(f"Error generating hypothetical document: {e}")
            document = f"ERROR: {str(e)}"
            total_tokens = 0

        generation_time = time.time() - start_time
        return document, generation_time, total_tokens

    async def generate_hypothetical_documents_batch(
        self,
        questions: List[str],
        num_docs_per_question: int = 1,
        concurrency: int = 10
    ) -> List[List[Tuple[str, float, int]]]:
        """
        Generate multiple hypothetical documents for each question.

        Args:
            questions: List of questions
            num_docs_per_question: Number of documents to generate per question
            concurrency: Maximum concurrent requests

        Returns:
            List of lists of (document, generation_time, total_tokens)
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def generate_with_semaphore(q):
            async with semaphore:
                return await self.generate_hypothetical_document(q)

        # Create tasks for all documents
        all_tasks = []
        for question in questions:
            question_tasks = [generate_with_semaphore(question) for _ in range(num_docs_per_question)]
            all_tasks.append(question_tasks)

        # Execute all tasks
        results = []
        for question_tasks in all_tasks:
            question_results = await asyncio.gather(*question_tasks)
            results.append(question_results)

        return results
