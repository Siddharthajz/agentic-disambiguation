"""
Integration Example: Judge in Agentic Pipeline

This shows how to integrate the LLM judge into the agentic disambiguation pipeline
for real-time answer quality assessment and optional regeneration.
"""

from typing import Optional, Dict, Any
from src.core import CachedJudge, OpenAIJudge, RAGConfig, AgentState
import logging

logger = logging.getLogger(__name__)


class IntegratedAgenticDisambiguation:
    """Extended agentic pipeline with LLM judge integration."""
    
    def __init__(self, config: RAGConfig, dataset: str = "ambignq"):
        # ... existing initialization ...
        
        # Add judge if requested
        if config.use_judge:
            logger.info(f"Initializing LLM Judge: {config.judge_model}")
            base_judge = OpenAIJudge(
                model=config.judge_model,
                temperature=0.0,
                api_key=config.openai_api_key
            )
            self.judge = CachedJudge(base_judge)
            self.quality_threshold = config.judge_quality_threshold
            self.max_regenerations = config.judge_max_regenerations
        else:
            self.judge = None
            logger.info("LLM Judge disabled")
    
    async def judge_answer_node(self, state: AgentState) -> AgentState:
        """
        LangGraph node for judging answer quality.
        
        Optional: Can trigger regeneration if quality is below threshold.
        """
        if not self.judge:
            return state
        
        try:
            question = state["question"]
            generated_answer = state["generated_answer"]
            ground_truth = state["reference_data"].get("all_short_answers", [])
            
            if not ground_truth:
                logger.debug("No ground truth available for judging")
                return state
            
            logger.info(f"Judging answer quality...")
            
            # Judge answer similarity
            judgment = await self.judge.judge_answer_similarity(
                question=question,
                generated_answer=generated_answer,
                ground_truth_answers=ground_truth
            )
            
            # Store judgment in metadata
            if not state.get("metadata"):
                state["metadata"] = {}
            
            state["metadata"]["judge_similarity"] = judgment
            similarity_score = judgment.get("similarity_score", 0.0)
            
            logger.info(f"Judge Similarity Score: {similarity_score:.3f}")
            
            # Optional: Judge disambiguation if ambiguous
            if state.get("is_ambiguous"):
                qa_pairs = state["reference_data"].get("qa_pairs", [])
                
                if len(qa_pairs) > 1:
                    logger.info(f"Judging disambiguation coverage ({len(qa_pairs)} interpretations)...")
                    
                    disambiguation_judgment = await self.judge.judge_disambiguation(
                        question=question,
                        generated_answer=generated_answer,
                        ground_truth_interpretations=qa_pairs,
                        dataset=self.dataset
                    )
                    
                    state["metadata"]["judge_disambiguation"] = disambiguation_judgment
                    
                    covered = disambiguation_judgment.get("interpretations_covered", 0)
                    total = disambiguation_judgment.get("total_interpretations", 0)
                    logger.info(f"Judge Disambiguation: {covered}/{total} interpretations covered")
            
            # Check if regeneration is needed (optional)
            if similarity_score < self.quality_threshold and state.get("regeneration_attempts", 0) < self.max_regenerations:
                logger.warning(f"Quality below threshold ({similarity_score:.3f} < {self.quality_threshold:.3f}). Triggering regeneration...")
                state["requires_regeneration"] = True
                state["regeneration_attempts"] = state.get("regeneration_attempts", 0) + 1
                state["judge_feedback"] = judgment.get("reasoning", "")
            
            return state
            
        except Exception as e:
            logger.error(f"Judge node error: {e}")
            state["metadata"]["judge_error"] = str(e)
            return state
    
    async def regenerate_answer_node(self, state: AgentState) -> AgentState:
        """
        Optional node to regenerate answer based on judge feedback.
        """
        if not state.get("requires_regeneration"):
            return state
        
        try:
            logger.info(f"Regenerating answer (attempt {state.get('regeneration_attempts', 1)})...")
            
            feedback = state.get("judge_feedback", "")
            original_answer = state.get("generated_answer", "")
            
            # Build regeneration prompt with judge feedback
            regeneration_prompt = f"""
            Your previous answer received low quality scores from an evaluator.
            
            Original Question: {state["question"]}
            Your Previous Answer: {original_answer}
            
            Evaluator Feedback: {feedback}
            
            Please provide a better, improved answer that addresses the feedback.
            """
            
            # Re-run synthesis with augmented prompt
            # (This would need to be integrated into synthesize_answer_node)
            
            logger.info("Regeneration complete")
            state["requires_regeneration"] = False
            
            return state
            
        except Exception as e:
            logger.error(f"Regeneration error: {e}")
            state["regeneration_error"] = str(e)
            return state
    
    def _build_workflow_with_judge(self):
        """Build LangGraph workflow with judge node integrated."""
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(AgentState)
        
        # Existing nodes
        workflow.add_node("detect_ambiguity", self.detect_ambiguity_node)
        workflow.add_node("generate_subqueries", self.generate_subqueries_node)
        workflow.add_node("generate_hyde", self.generate_hyde_docs_node)
        workflow.add_node("retrieve_hyde", self.retrieve_with_hyde_node)
        workflow.add_node("simple_retrieval", self.simple_retrieval_node)
        workflow.add_node("synthesize_answer", self.synthesize_answer_node)
        
        # NEW: Judge node
        if self.judge:
            workflow.add_node("judge_answer", self.judge_answer_node)
            workflow.add_node("regenerate_answer", self.regenerate_answer_node)
        
        # Set entry point
        workflow.set_entry_point("detect_ambiguity")
        
        # Conditional edge after ambiguity detection
        workflow.add_conditional_edges(
            "detect_ambiguity",
            self.should_decompose,
            {
                "generate_subqueries": "generate_subqueries",
                "simple_retrieval": "simple_retrieval",
            }
        )
        
        # Edges for ambiguous path
        workflow.add_edge("generate_subqueries", "generate_hyde")
        workflow.add_edge("generate_hyde", "retrieve_hyde")
        workflow.add_edge("retrieve_hyde", "synthesize_answer")
        
        # Edge for simple path
        workflow.add_edge("simple_retrieval", "synthesize_answer")
        
        # NEW: Judge edges
        if self.judge:
            workflow.add_edge("synthesize_answer", "judge_answer")
            
            # Conditional edge: regenerate if needed, otherwise finish
            workflow.add_conditional_edges(
                "judge_answer",
                lambda state: "regenerate_answer" if state.get("requires_regeneration") else "finish",
                {
                    "regenerate_answer": "regenerate_answer",
                    "finish": END
                }
            )
            
            # Regeneration loops back to judge
            workflow.add_edge("regenerate_answer", "judge_answer")
        else:
            # Original flow: end after synthesis
            workflow.add_edge("synthesize_answer", END)
        
        return workflow.compile()


# ============================================================================
# Configuration Example
# ============================================================================

def example_config_with_judge() -> Dict[str, Any]:
    """Example configuration with judge enabled."""
    return {
        # Existing settings
        "retrieval_mode": "hybrid",
        "llm_model": "gpt-4o-mini",
        "top_k": 5,
        
        # Judge settings (NEW)
        "use_judge": True,
        "judge_model": "gpt-4o-mini",
        "judge_quality_threshold": 0.6,  # Regenerate if score < 0.6
        "judge_max_regenerations": 2,    # Max 2 attempts
    }


# ============================================================================
# CLI Integration Example
# ============================================================================

def add_judge_arguments_to_parser(parser):
    """Add judge-related arguments to argument parser."""
    judge_group = parser.add_argument_group('Judge Settings')
    
    judge_group.add_argument(
        "--use-judge",
        action="store_true",
        help="Enable LLM judge for answer quality assessment"
    )
    
    judge_group.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for judging (default: gpt-4o-mini)"
    )
    
    judge_group.add_argument(
        "--judge-quality-threshold",
        type=float,
        default=0.6,
        help="Quality threshold for regeneration (0-1, default: 0.6)"
    )
    
    judge_group.add_argument(
        "--judge-max-regenerations",
        type=int,
        default=2,
        help="Max regeneration attempts (default: 2)"
    )


# ============================================================================
# Output Example
# ============================================================================

def example_result_with_judge() -> Dict[str, Any]:
    """Example RAG result with judge metadata."""
    return {
        "question": "When was the US break away from England?",
        "question_id": "ambignq_001",
        "generated_answer": "The United States declared independence on July 4, 1776.",
        "retrieval_time": 0.25,
        "generation_time": 1.5,
        "total_tokens": 450,
        
        # Judge metadata (NEW)
        "metadata": {
            "ambiguity_status": "Ambiguous",
            "subqueries": [
                "When was the Declaration of Independence?",
                "When did the Revolutionary War end?",
                "When was the Treaty of Paris signed?"
            ],
            
            # Judge scores
            "judge_similarity": {
                "similarity_score": 0.85,
                "is_similar": True,
                "matched_answer": "July 4, 1776",
                "reasoning": "Generated answer directly matches one ground truth",
                "coverage_score": 0.9,
                "generation_time": 0.45,
                "total_tokens": 156
            },
            
            "judge_disambiguation": {
                "disambiguation_score": 0.67,
                "interpretations_covered": 2,
                "total_interpretations": 3,
                "covered_interpretations": [0, 1],
                "missing_interpretations": [2],
                "reasoning": "Covered Declaration and War end, missed Treaty"
            }
        },
        
        # Evaluation scores
        "evaluation": {
            "f1": 0.85,
            "d_f1": 0.67,
            "judge_similarity": 0.85,
            "judge_disambiguation": 0.67
        }
    }


# ============================================================================
# Usage in Evaluation
# ============================================================================

async def example_batch_evaluation_with_judge():
    """Example batch evaluation using judge."""
    from src.core import RAGConfig
    
    # Configuration
    config = RAGConfig()
    config.use_judge = True
    config.judge_model = "gpt-4o-mini"
    
    # Initialize pipeline with judge
    framework = IntegratedAgenticDisambiguation(config, dataset="ambignq")
    
    # Load test data
    import json
    with open("data/ambignq_test.json") as f:
        test_data = json.load(f)
    
    # Run pipeline (includes judge evaluation)
    results = await framework.run_batch(test_data, limit=10)
    
    # Extract judge scores for analysis
    judge_scores = []
    for result in results:
        judge_info = result.metadata.get("judge_similarity")
        if judge_info:
            judge_scores.append({
                "question_id": result.question_id,
                "similarity_score": judge_info.get("similarity_score"),
                "coverage_score": judge_info.get("coverage_score")
            })
    
    # Compute averages
    if judge_scores:
        avg_similarity = sum(s["similarity_score"] for s in judge_scores) / len(judge_scores)
        avg_coverage = sum(s["coverage_score"] for s in judge_scores) / len(judge_scores)
        
        print(f"\nJudge Evaluation Results:")
        print(f"  Average Similarity: {avg_similarity:.3f}")
        print(f"  Average Coverage: {avg_coverage:.3f}")
        print(f"  Scores saved to judge_scores.json")
        
        with open("judge_scores.json", "w") as f:
            json.dump(judge_scores, f, indent=2)


if __name__ == "__main__":
    print("Integration Example: Judge in Agentic Pipeline")
    print("\nKey integration points:")
    print("1. judge_answer_node() - Evaluate answer quality")
    print("2. regenerate_answer_node() - Optional regeneration")
    print("3. _build_workflow_with_judge() - Updated LangGraph workflow")
    print("4. add_judge_arguments_to_parser() - CLI integration")
    print("\nSee docs/JUDGE_GUIDE.md for complete documentation")
