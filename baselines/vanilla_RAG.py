import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import openai
import tiktoken
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import AutoQueryEncoder
from tqdm.asyncio import tqdm as async_tqdm 

# Import evaluation module
from evaluation import RAGEvaluator, print_evaluation_report


@dataclass
class RetrievalResult:
    """Stores a single retrieval result."""
    doc_id: str
    title: str
    text: str
    score: float
    rank: int
    source: str = "sparse"  # "sparse", "dense", or "hybrid"


@dataclass
class RAGResult:
    """Stores the result of a RAG pipeline run."""
    question_id: str
    question: str
    retrieved_docs: List[Dict[str, Any]]
    generated_answer: str
    reference_data: Dict[str, Any]
    retrieval_time: float
    generation_time: float
    total_tokens: int
    evaluation: Dict[str, Any]


class CleanRAG:
    """Clean RAG implementation with sparse, dense, and hybrid retrieval."""

    def __init__(
        self,
        retrieval_mode: str = "sparse",
        sparse_index: str = "wikipedia-dpr",
        dense_index: str = "wikipedia-dpr-100w.bpr-single-nq",
        dense_encoder: str = "castorini/bpr-nq-question-encoder",
        llm_model: str = "gpt-4.1-mini",
        top_k: int = 5,
        max_tokens: int = 200,
        temperature: float = 0.0,
        hybrid_alpha: float = 0.5
    ):
        self.retrieval_mode = retrieval_mode
        self.llm_model = llm_model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.hybrid_alpha = hybrid_alpha

        self.sparse_searcher = None
        self.dense_searcher = None

        if retrieval_mode in ["sparse", "hybrid"]:
            print(f"Loading sparse retriever (BM25): {sparse_index}...")
            self.sparse_searcher = LuceneSearcher.from_prebuilt_index(sparse_index, True)
            print(f"✓ Sparse retriever loaded")

        if retrieval_mode in ["dense", "hybrid"]:
            print(f"Loading dense retriever (FAISS): {dense_index}...")
            print(f"Loading query encoder: {dense_encoder}...")
            query_encoder = AutoQueryEncoder(
                encoder_dir=dense_encoder, pooling='mean', l2_norm=True
            )
            self.dense_searcher = FaissSearcher.from_prebuilt_index(
                dense_index, query_encoder
            )
            print(f"✓ Dense retriever loaded")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.async_client = openai.AsyncOpenAI(api_key=api_key)

        try:
            self.tokenizer = tiktoken.encoding_for_model(llm_model)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.evaluator = RAGEvaluator(k=top_k, d_f1_threshold=0.5)

    def retrieve_sparse(self, query: str, k: int) -> List[RetrievalResult]:
        """Retrieve using BM25 (sparse retrieval)."""
        hits = self.sparse_searcher.search(query, k=k)
        results = []
        for rank, hit in enumerate(hits, 1):
            try:
                doc = self.sparse_searcher.doc(hit.docid)
                doc_dict = json.loads(doc.raw())
                results.append(RetrievalResult(
                    doc_id=hit.docid,
                    title=doc_dict.get('title', 'Unknown'),
                    text=doc_dict.get('contents', ''),
                    score=hit.score,
                    rank=rank,
                    source="sparse"
                ))
            except Exception as e:
                print(f"Warning: Failed to parse document {hit.docid}: {e}")
        return results

    def retrieve_dense(self, query: str, k: int) -> List[RetrievalResult]:
        """Retrieve using FAISS (dense retrieval)."""
        hits = self.dense_searcher.search(query, k=k)
        results = []
        for rank, hit in enumerate(hits, 1):
            try:
                doc = self.dense_searcher.doc(hit.docid)
                doc_dict = json.loads(doc.raw())
                results.append(RetrievalResult(
                    doc_id=hit.docid,
                    title=doc_dict.get('title', 'Unknown'),
                    text=doc_dict.get('contents', ''),
                    score=hit.score,
                    rank=rank,
                    source="dense"
                ))
            except Exception as e:
                print(f"Warning: Failed to parse document {hit.docid}: {e}")
        return results

    def retrieve_hybrid(self, query: str, k: int) -> List[RetrievalResult]:
        """Hybrid retrieval using Reciprocal Rank Fusion (RRF)."""
        sparse_results = self.retrieve_sparse(query, k=k)
        dense_results = self.retrieve_dense(query, k=k)

        # Create a dictionary to store RRF scores for each document
        rrf_scores = {}
        # A map to get the full result object from a doc_id
        doc_map = {res.doc_id: res for res in sparse_results + dense_results}
        
        # Process sparse results
        for rank, res in enumerate(sparse_results, 1):
            rrf_scores[res.doc_id] = rrf_scores.get(res.doc_id, 0) + 1 / (60 + rank)

        # Process dense results
        for rank, res in enumerate(dense_results, 1):
            rrf_scores[res.doc_id] = rrf_scores.get(res.doc_id, 0) + 1 / (60 + rank)
            
        # Sort documents by their combined RRF score
        sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Create final results list
        results = []
        for rank, doc_id in enumerate(sorted_docs[:k], 1):
            original_result = doc_map[doc_id]
            results.append(RetrievalResult(
                doc_id=doc_id,
                title=original_result.title,
                text=original_result.text,
                score=rrf_scores[doc_id], 
                rank=rank,
                source="hybrid"
            ))
        return results

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Main retrieval method that dispatches to appropriate retriever."""
        if self.retrieval_mode == "sparse":
            return self.retrieve_sparse(query, self.top_k)
        elif self.retrieval_mode == "dense":
            return self.retrieve_dense(query, self.top_k)
        elif self.retrieval_mode == "hybrid":
            return self.retrieve_hybrid(query, self.top_k)
        else:
            raise ValueError(f"Invalid retrieval mode: {self.retrieval_mode}")

    async def generate_async(self, question: str, contexts: List[RetrievalResult]) -> Tuple[str, float, int]:
        """Generate answer asynchronously for a single question."""
        context_str = "\n\n".join([
            f"Document {ctx.rank} (Title: {ctx.title}):\n{ctx.text[:500]}"
            for ctx in contexts
        ])
        prompt = f"""Using the context below, answer the question **only** with the final answer.
        - Be concise and factual.
        - Do not include explanations, reasoning, or extra text.
        - Respond in a single line suitable for automatic comparison with ground truth.

        Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"""

        start_time = time.time()
        try:
            response = await self.async_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            answer = response.choices[0].message.content.strip()
            total_tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
        except Exception as e:
            print(f"Error during async generation: {e}")
            answer, total_tokens = f"ERROR: {str(e)}", 0
        
        generation_time = time.time() - start_time
        return answer, generation_time, total_tokens

    async def run_single_async(self, question: str, question_id: str, reference_data: Dict) -> RAGResult:
        """Run the RAG pipeline on a single question asynchronously with full evaluation."""
        start_time = time.time()
        retrieved_docs = self.retrieve(question)
        retrieval_time = time.time() - start_time

        answer, generation_time, total_tokens = await self.generate_async(question, retrieved_docs)

        evaluation = self.evaluator.evaluate_single(
            prediction=answer,
            retrieved_docs=[asdict(doc) for doc in retrieved_docs],
            reference_item=reference_data,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_tokens=total_tokens
        )
        return RAGResult(
            question_id=question_id,
            question=question,
            retrieved_docs=[asdict(doc) for doc in retrieved_docs],
            generated_answer=answer,
            reference_data=reference_data,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_tokens=total_tokens,
            evaluation=evaluation
        )
    
    async def _process_item_with_semaphore(self, item: Dict, semaphore: asyncio.Semaphore) -> RAGResult:
        """Wrapper to process a single item with a semaphore for concurrency control."""
        question = item['question']
        question_id = item.get('id', str(hash(question)))
        async with semaphore:
            try:
                return await self.run_single_async(question, question_id, item)
            except Exception as e:
                print(f"\nError processing question '{question}': {e}")
                return RAGResult(
                    question_id=question_id, question=question, retrieved_docs=[],
                    generated_answer=f"ERROR: {str(e)}", reference_data=item,
                    retrieval_time=0.0, generation_time=0.0, total_tokens=0, evaluation={}
                )

    async def run_experiment_async(
        self,
        test_data: List[Dict],
        limit: Optional[int] = None,
        concurrency: int = 10
    ) -> List[RAGResult]:
        """Run the RAG pipeline concurrently on the entire test set."""
        if limit:
            test_data = test_data[:limit]

        semaphore = asyncio.Semaphore(concurrency)
        tasks = [self._process_item_with_semaphore(item, semaphore) for item in test_data]
        
        # Use async_tqdm for a progress bar that works with asyncio
        results = await async_tqdm.gather(
            *tasks, desc=f"Running RAG ({self.retrieval_mode}, Concurrency: {concurrency})"
        )
        return results

def main():
    parser = argparse.ArgumentParser(description="Clean RAG baseline with sparse, dense, and hybrid retrieval")
    parser.add_argument("--data-path", type=str, default="data/ambignq_test.json", help="Path to AmbigNQ test data")
    parser.add_argument("--output-path", type=str, default="results/rag_results.json", help="Path to save results")
    parser.add_argument("--retrieval-mode", type=str, default="all", choices=["sparse", "dense", "hybrid", "all"], help="Retrieval mode to run")
    parser.add_argument("--sparse-index", type=str, default="wikipedia-dpr", help="PySerini sparse index name")
    parser.add_argument("--dense-index", type=str, default="wikipedia-dpr-100w.bpr-single-nq", help="PySerini FAISS index name")
    parser.add_argument("--dense-encoder", type=str, default="castorini/bpr-nq-question-encoder", help="Query encoder for FAISS")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens for generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test examples")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent OpenAI requests")
    
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    print(f"Loading test data from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test examples")

    modes_to_run = ["sparse", "dense", "hybrid"] if args.retrieval_mode == "all" else [args.retrieval_mode]
    all_results = {}

    for mode in modes_to_run:
        print(f"\n{'='*60}\n  Running {mode.upper()} retrieval mode\n{'='*60}\n")
        rag = CleanRAG(
            retrieval_mode=mode, sparse_index=args.sparse_index,
            dense_index=args.dense_index, dense_encoder=args.dense_encoder,
            llm_model=args.model, top_k=args.top_k, max_tokens=args.max_tokens
        )
        
        # Directly call and run the async method
        results = asyncio.run(rag.run_experiment_async(test_data, limit=args.limit, concurrency=args.concurrency))
        
        print(f"\nComputing aggregate metrics for {mode}...")
        aggregate_metrics = rag.evaluator.evaluate_batch([asdict(r) for r in results])
        
        print(f"\n{'='*60}\n  Results for {mode.upper()} mode\n{'='*60}")
        print_evaluation_report(aggregate_metrics)
        
        all_results[mode] = {
            "config": vars(args),
            "aggregate_metrics": aggregate_metrics,
            "results": [asdict(r) for r in results]
        }
        
        if args.retrieval_mode == "all":
            print(f"\n🧹 Cleaning up {mode} retriever to free memory...")
            del rag
            import gc
            gc.collect()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.retrieval_mode == "all":
        for mode, data in all_results.items():
            mode_output = output_path.parent / f"{output_path.stem}_{mode}{output_path.suffix}"
            with open(mode_output, 'w') as f: json.dump(data, f, indent=2)
            print(f"\n{mode.upper()} results saved to {mode_output}")
    else:
        with open(output_path, 'w') as f: json.dump(all_results[args.retrieval_mode], f, indent=2)
        print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()