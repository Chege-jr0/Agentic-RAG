"""
The self correction engine.
This is what makes our RAG truly agentic. After retrieving chunks the agent doesn't just blindly
generate an answer, it stops and asks itself, do I actually have enough information to answer this well.

If the answer is no, it tries again with a better query, this is called self-correction loop.
This loop can only run 3 times.
"""

from dataclasses import dataclass
from retriever import RetrievedChunk

@dataclass
class EvaluationResult:
    """
    Result of context evaluation.
    Contains the decision and reasoning behind it.
    """

    is_sufficient: bool
    confidence: float # 0.0 to 1.0
    reasoning: float # why sufficient or not
    missing_aspects: list[str] #whats missing if not sufficient
    coverage_score: float # how much of query is covered
    attempt_number: int # which retrieval attempts this is
    recommendation: str # what to do next

class ContextEvaluator:
    """
    Evaluates whether retrived chunks contain sufficient context to answer
    the user's query well.

    Uses heuristics scoring - fast and reliable 
    No LLM needed for this step.
    """ 

    # Minimum thresholds for sufficiency
    MIN_CHUNKS = 1 # Need at least 1 chunk
    MIN_COVERAGE_SCORE = 0.35 # Need 35% query word coverage
    MIN_TOP_SCORE = 0.25 # Top chunk must score above this
    MAX_ATTEMPTS = 3  # Maximu retrieval attempts

    def __init__(self):
        print("Evaluator ready!")

    # Calculating the average score
    def calculate_coverage( self, query: str, chunks: list[RetrievedChunk]) -> float:
        """
        Calculate how much of the query is covered by the retrieved chunks.

        Coverage = overlap between query words and chunk words
        """   
        if not chunks:
            return 0.0
        
        # Get meaning query words
        stop_words = {
             "the", "a", "an", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "of", "and",
            "or", "but", "with", "what", "how", "why",
            "which", "who", "where", "when", "does", "do",
            "tell", "me", "about", "explain", "describe"
        }

        query_words = set(query.lower().split()) - stop_words

        if not query_words:
            return 0.5 # Can't evaluate - assume moderate average
        
        # Combine all chunk text
        all_chunk_text = " ".join(
            c.text.lower() for c in chunks
        )
        

        chunk_words = set(all_chunk_text.split())

        # Calculate overlap
        overlap = query_words & chunk_words
        coverage = len(overlap) / len(query_words)

        return round(coverage, 3)
    
    def find_missing_aspects(self, query: str, chunks: list[RetrievedChunk]) -> list[str]:
        """
        Identify which aspects of the query are not covered by the retrieved chunks

        """
        missing = []
        query_lower = query.lower()
        all_chunk_text = " ".join(
            c.text.lower() for c in chunks
        )

        # Check for specific aspects that might be missing
        aspect_checks = {
            "gender": ["gender", "male", "female", "women", "men"],
            "county comparison": ["county", "counties", "region"],
            "time period": ["2024", "2023", "2022", "2021", "2006"],
            "barriers": ["barrier", "obstacle", "challenge", "reason"],
            "products": ["mpesa", "m-pesa", "fuliza", "shwari", "sacco"],
            "urban rural": ["urban", "rural", "city", "village"],
            "statistics": ["percent", "%", "rate", "number", "million"]
        }

        for aspect, keywords in aspect_checks.items():
            # Check if query asks about this aspect
            query_asks = any(k in query_lower for k in keywords)
            # Check if chunks cover this aspect
            chunks_cover = any(k in all_chunk_text for k in keywords)

            if query_asks and not chunks_cover:
                missing.append(aspect)

        return missing


    def evaluate(self, query: str, chunks: list[RetrievedChunk], attempt_number: int = 1) -> EvaluationResult:
        """
        Main Evalutaion function - called by the agent.
        Decides if retrieved context is sufficient to answer
        """     

        print(f"Evaluating context (attempt {attempt_number}/{self.MAX_ATTEMPTS})...")  

        #No chunks at all
        if not chunks:
            return EvaluationResult(
                is_sufficient=False,
                confidence=0.95,
                reasoning="No chunks retrieved - retrieval failed completely",
                missing_aspects=["all information"],
                coverage_score=0.0,
                attempt_number=attempt_number,
                recommendation="retry_with_different_query"
            ) 
        
        # Calculate Metrics
        coverage_score = self.calculate_coverage(query, chunks)
        missing_aspects = self.find_missing_aspects(query, chunks)
        top_score = max(c.score for c in chunks)
        avg_score = sum(c.score for c in chunks) / len(chunks)
        num_chucks = len(chunks)

        # Decision logic
        sufficient = True
        reasoning_parts = []
        confidence = 0.8

        # Check minimum chunks
        if num_chucks < self.MIN_CHUNKS:
            sufficient = False
            reasoning_parts.append(
                f"Only {num_chucks} chunks retrieved => need more"
            )

        # Check top score
        if top_score < self.MIN_TOP_SCORE:
            sufficient = False
            reasoning_parts.append(
                f"Best chunk score {top_score:.2f} below threshold {self.MIN_TOP_SCORE}"
            ) 

        # Check coverage
        if coverage_score < self.MIN_COVERAGE_SCORE:
            sufficient = False
            reasoning_parts.append(
                f"Coverage {coverage_score:.1%} below minimum {self.MIN_COVERAGE_SCORE:.1%}"
            )    
            confidence -= 0.1

        # Missing critical aspects
        if len(missing_aspects) > 2:
            sufficient = False
            reasoning_parts.append(
                f"Missing key aspects: {','.join(missing_aspects[:3])}"
            )    
            confidence -= 0.1

        # On final attempt always proceed
        if attempt_number >= self.MAX_ATTEMPTS:
            sufficient = True
            reasoning_parts.append(
                "Maximum attempts reached -  proceeding with best available context"
            )    
            confidence = 0.5

        if sufficient and not reasoning_parts:
            reasoning = (
                f"Context sufficient - coverage: {coverage_score:.1%}, "
                f"top score: {top_score:.2f}, chunks: {num_chucks}"
            )   
        else:
            reasoning = " | ".join(reasoning_parts) if reasoning_parts else "Sufficient"

        if sufficient:
            recommendation = "proceed_to_generation"

        elif attempt_number < self.MAX_ATTEMPTS:
            recommendation = "retry_with_expanded_query"
        else:
            recommendation = "proceed_with_partial_context"

        result = EvaluationResult(
            is_sufficient=sufficient,
            confidence=round(confidence, 2),
            reasoning=reasoning,
            missing_aspects=missing_aspects,
            coverage_score=coverage_score,
            attempt_number=attempt_number,
            recommendation=recommendation
        )    

        #Print summary
        status = "Sufficient" if sufficient else "Insufficient"
        print(f"{status} | coverage = { coverage_score:.1%} | "
              f"top_score = {top_score:.2f} | chunks={num_chucks}")
        if not sufficient:
            print(f"Recommendation: {recommendation}")

        return result                 

if __name__ == "__main__":
    from retriever import HybridRetriever, RetrievedChunk
    from reranker import Reranker

    print("Testing ContextEvaluator...")

    retriever = HybridRetriever()
    reranker = Reranker()
    evaluator = ContextEvaluator()

    test_cases = [
        {
            "query": "What is Kenya's financial inclusion rate?",
            "description": "Should be SUFFICIENT - direct match"
        },
        {
            "query": "What is GDP of Kenya in 2024",
            "description": "Should be INSUFFICIENT - not in our documents"
            
        },
        {
            "query": "Explain the gender gap and urban rural divide",
            "description": "Should be SUFFICIENT - both covered"
        }
    ]
    for case in test_cases:
        query = case["query"]
        print(f"Query: {query}")
        print(f"Expected: {case['description']}")

        chunks, strategy = retriever.search(query, k=5)
        final_chunks, _ = reranker.process(query, chunks)

        result = evaluator.evaluate(query, final_chunks, attempt_number=1)

        print(f"Decsion: {'SUFFICIENT' if result.is_sufficient else 'INSUFFICIENT'}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Next step: {result.recommendation}")
        if result.missing_aspects:
            print(f"Missing: {result.missing_aspects}")




