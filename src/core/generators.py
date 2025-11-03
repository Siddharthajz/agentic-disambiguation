"""
Generator components for RAG systems.

Provides modular generator implementations:
- BaseGenerator: Abstract base class
- OpenAIGenerator: OpenAI API implementation with async batching
- LlamaCppGenerator: Local LLM inference via llama.cpp (no API required)
"""

import asyncio
import logging
import time
import threading
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import openai
import tiktoken

from .data_models import RetrievalResult

logger = logging.getLogger(__name__)

# Optional import for local LLM support
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. Local LLM support unavailable.")


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
- Respond with only the answer for automatic comparison with ground truth.
- Do not state "Based on the context" or similar phrases, not even restating the question. Just provide the answer.

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


class LlamaCppGenerator(BaseGenerator):
    """
    Local LLM generator using llama.cpp.

    Runs models locally without API dependency. Suitable for M1 Macs with
    Metal acceleration. Supports GGUF quantized models.

    Default parameters optimized for Qwen3-4B-Instruct-2507:
    - temperature: 0.7 (Qwen recommended)
    - top_p: 0.8 (Qwen recommended)
    - top_k: 20 (Qwen recommended)
    - n_ctx: 8192 (reasonable for RAG with retrieved docs)
    """

    def __init__(
        self,
        model_path: str,
        max_tokens: int = 200,
        temperature: float = 0.7,  # Qwen recommended: 0.7
        n_ctx: int = 8192,  # Reasonable for RAG pipeline (5 docs × ~500 tokens + prompt)
        n_gpu_layers: int = -1,  # -1 = offload all layers to GPU
        top_p: float = 0.8,  # Qwen recommended: 0.8
        top_k: int = 20,  # Qwen recommended: 20
        repeat_penalty: float = 1.1,  # Slight penalty to reduce repetition
        verbose: bool = False
    ):
        """
        Initialize llama.cpp generator.

        Args:
            model_path: Path to GGUF model file
            max_tokens: Max tokens for generation
            temperature: Sampling temperature (Qwen recommends 0.7)
            n_ctx: Context window size (default 8192 for RAG)
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            top_p: Top-p sampling (Qwen recommends 0.8)
            top_k: Top-k sampling (Qwen recommends 20)
            repeat_penalty: Repetition penalty (1.1 for slight reduction)
            verbose: Enable verbose llama.cpp logging
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            )

        super().__init__(model=model_path, max_tokens=max_tokens, temperature=temperature)

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty

        logger.info(f"Loading local LLM from {model_path}...")
        logger.info(f"GPU layers: {n_gpu_layers}, Context size: {n_ctx}")
        logger.info(f"Sampling params: temp={temperature}, top_p={top_p}, top_k={top_k}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            n_threads=8,
            n_batch=512,
            flash_attn=True,
            use_mlock=False,
            use_mmap=True
        )

        self._lock = threading.Lock()

        logger.info("Local LLM loaded successfully")

    def _generate_sync(self, prompt: str, system_prompt: str = None, temperature: float = None, top_p: float = None, top_k: int = None) -> Tuple[str, int]:
        """
        Synchronous generation helper with thread safety.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override (uses self.temperature if None)
            top_p: Optional top_p override (uses self.top_p if None)
            top_k: Optional top_k override (uses self.top_k if None)

        Returns:
            Tuple of (generated_text, token_count)
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Acquire lock to ensure thread-safe access to llama.cpp
        with self._lock:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                top_p=top_p if top_p is not None else self.top_p,
                top_k=top_k if top_k is not None else self.top_k,
                repeat_penalty=self.repeat_penalty,
                stop=["</s>", "<|im_end|>", "<|endoftext|>"]
            )

            generated_text = response["choices"][0]["message"]["content"].strip()
            total_tokens = response["usage"]["total_tokens"]

        return generated_text, total_tokens

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
        # Use the same format_prompt method as OpenAI generator for consistency
        prompt = self.format_prompt(question, contexts)

        # Clear system prompt - instructions are in the user prompt
        system_prompt = None

        start_time = time.time()
        try:
            # Run in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            answer, tokens = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt,
                system_prompt
            )
        except Exception as e:
            logger.error(f"Error generating answer with local LLM: {e}")
            answer = f"ERROR: {str(e)}"
            tokens = 0

        generation_time = time.time() - start_time
        return answer, generation_time, tokens

    async def generate_hypothetical_document(self, question: str) -> Tuple[str, float, int]:
        """
        Generate a hypothetical document for HyDE retrieval.

        Args:
            question: Question to generate document for

        Returns:
            Tuple of (document, generation_time, total_tokens)
        """
        prompt = f"""Write a brief, factual passage that would appear in Wikipedia and contain the answer to this question. Output ONLY the passage, with no preamble or explanation.

Question: {question}

Passage:"""

        # No system prompt - instructions are in user prompt for consistency
        system_prompt = None

        start_time = time.time()
        try:
            loop = asyncio.get_event_loop()
            # Use higher temperature (0.7) for HyDE generation diversity
            document, tokens = await loop.run_in_executor(
                None,
                lambda: self._generate_sync(prompt, system_prompt, temperature=0.7)
            )
        except Exception as e:
            logger.error(f"Error generating HyDE document with local LLM: {e}")
            document = f"ERROR: {str(e)}"
            tokens = 0

        generation_time = time.time() - start_time
        return document, generation_time, tokens
