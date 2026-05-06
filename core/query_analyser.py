"""
The Query Analyser file purpose:
The first thing the agent does when it receives the question, It classifies the question before anything else
Simple Question: What is Kenya's inclusion rate?
This is a direct retrieval - direct answer

Complex Question: Compare Turkana and Kimabu inclusion rates and explain why they are different?
This is multiple retrievals and then structured reasoning.

Multi-hop Question: Which country has the highest inclusion and lowest barrier to entry and what product should 
someone there use first?
This is decomposed into 3 sub-questions and then chain answers
"""

from dataclasses import dataclass
from enum import Enum

#Enum creates a fixed set of named constants, Instead of using simple strings like "simple" we use QueryType.SIMPLE

class QueryType(Enum):
    """Classification of query complexity."""
    SIMPLE = "simple"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"

class RetrievalStrategy(Enum):
    """Recommended retrieval strategy per query type."""
    DIRECT = "direct"  # One retrieval pass
    EXPANDED = "expanded" # Retrieve more chunks
    DECOMPOSED = "decomposed" # Break into sub-questions

@dataclass
class QueryAnalysis:
    """
    Complete analysis of a user query
    Contains everything the agent needs to plan its approach
    """   
    original_query: str
    query_type: QueryType
    retrieval_strategy: RetrievalStrategy
    complexity_score: float #0.0 to 1.0
    requires_comparison: bool # "compare X and Y"
    requires_reasoning: bool # why / explain / analyze 
    requires_multi_step: bool # mutiple facts needed
    sub_questions: list[str] # decomposed questions
    confidence: float # how confident in classification
    reasoning: str # why this classification


class QueryAnalyser:
    """
    Analyses user queries to determin complexity,
    required retrieval strategy and whether decomposition is needed.

    Uses rule-based heuristics - no LLM needed for this step.
    Fast and deterministic.

    LLM classification adds latency and can be inconsistent.
    For classification, rule works very well.
    """

    #Words that signal complexity
    COMPARISON_WORDS = {
        "compare", "versus", "vs", "difference", "between",
        "better", "worse", "higher", "lower", "more", "less",
        "contrast", "similar", "different"
    }

    REASONING_WORDS = {
        "why", "explain", "reason", "cause", "because",
        "analyze", "analyse", "understand", "how does",
        "what drives", "what causes", "impact", "effect",
        "influence", "relationship"
    }

    MULTI_HOP_SIGNALS = {
        "and", "also", "additionally", "furthermore",
        "as well as", "both", "all", "which county",
        "what product", "recommend", "should i", "best for"
    }

    SIMPLE_STARTERS = {
        "what is", "what are", "when", "where",
        "who", "how many", "how much", "what was",
        "what percentage", "what rate", "define", "list"
    }

    def check_comparison(self, query: str) -> bool:
        """
        Check if thr query reuires comparing two or more things.
        """
        query_lower = query.lower()
        return any(word in query_lower for word in self.COMPARISON_WORDS)
    
    
    def check_reasoning(self, query: str) -> bool:
        """
        Check if query requiress explanation or reasoning.
        """
        query_lower = query.lower()
        return any(word in query_lower for word in self.REASONING_WORDS)
    
    
    def check_multi_hop(self, query: str) -> bool:
        """
        Check if query requires chaining mutiple facts.
        """
        query_lower = query.lower()
        word_count = len(query_lower.split())

        # Multi-hop signals present
        has_signals = any(
            signal in query_lower
            for signal in self.MULTI_HOP_SIGNALS
        )

        # Long queries are more likely to multi-hop
        is_long = word_count > 15

        # Multiple question marks or "and" suggest multi-part
        has_mutiple_parts = query_lower.count(" and ") >=2

        return (has_signals and is_long) or has_mutiple_parts
    
    def calculate_complexity(self, query: str) -> float:
        """
        Calculate a complexity score from 0.0 to 1.0.
        Higher = more complex = needs more careful handling.
        """
        score = 0.0
        query_lower = query.lower()
        words = query_lower.split()

        # Word count contribution (longer = more complex)
        if len(words) > 20:
            score += 0.3
        elif len(words) > 10:
            score += 0.15    

        # Comparison adds complexity
        if self.check_comparison(query):
            score += 0.25

        # Reasoning adds complexity
        if self.check_reasoning(query):
            score += 0.25

        # Multi-hop adds most complexity
        if self.check_multi_hop(query):
            score += 0.3

        # Simple starters reduce complexity
        if any(query_lower.startswith(s) for s in self.SIMPLE_STARTERS):
            score -= 0.2

        return round(min(1.0, max(0.0, score)), 2)  

    def decompose_query(self, query: str) -> list[str]:
        """
        Decompose a complex multi-hop query into sub-questions.
        Each sub-question can be answered indepedently
        """  

        sub_questions = []
        query_lower = query.lower()

        # Split on "and" for compound questions
        if " and " in query_lower:
            parts = query.split(" and ")               
            for part in parts:
                part = part.strip()
                if len(part.split()) >= 3:
                    #Make sure it reads as a question
                    if not part.endswith("?"):
                        part += "?"
                    sub_questions.append(part)    

        # If no decomposition found use original
        if not sub_questions:
            sub_questions = [query]

        return sub_questions

    def analyse(self, query: str) -> QueryAnalysis:
        """
        Main analysis function - called by the agent.
        Returns complete QueryAnalysis with all decisions made.
        """   

        query_lower = query.lower()
        words = query_lower.split()

        #Check each dimension
        requires_comparison = self.check_comparison(query)
        requires_reasoning = self.check_reasoning(query)
        requires_multi_step = self.check_multi_hop(query)
        complexity_score = self.calculate_complexity(query)

        # Classify query type
        if complexity_score <= 0.2:
            query_type = QueryType.SIMPLE
            retrieval_strategy = RetrievalStrategy.DIRECT
            reasoning = "Short, direct factual question"
            confidence = 0.90

        elif requires_multi_step or complexity_score >= 0.6:
            query_type = QueryType.MULTI_HOP    
            retrieval_strategy = RetrievalStrategy.DECOMPOSED
            reasoning = "Multiple facts needed - decomposing into sub-questions"
            confidence = 0.82

        else:
            query_type = QueryType.COMPLEX
            retrieval_strategy = RetrievalStrategy.EXPANDED
            reasoning = "Complex question requiring broader contecxt"
            confidence = 0.85

        # Decompose if multi-hop
        sub_questions = []
        if query_type == QueryType.MULTI_HOP:
            sub_questions = self.decompose_query(query)  

        return QueryAnalysis(
            original_query = query,
            query_type = query_type,
            retrieval_strategy = retrieval_strategy,
            complexity_score = complexity_score,
            requires_comparison = requires_comparison,
            requires_reasoning = requires_reasoning,
            requires_multi_step = requires_multi_step,
            sub_questions = sub_questions,
            confidence = confidence,
            reasoning =  reasoning

        )   

if __name__ == "__main__":
    print("Testing QueryAnalyser...")

    analyser = QueryAnalyser()

    test_queries = [
        "What is Kenya's financial inclusion rate?",
        "What is M-Pesa?"
        "Compare the gender gap in 2006 and 2024 and explain what caused the change and recommend what policy should focus on next",
        "What is M-Pesa?",
        "Analyze the relationship between mobile phone access and financial exclusion in rural Kenya"
    ]   

    for query in test_queries:
        print(f"Query: {query[:60]}....")
        analysis = analyser.analyse(query)

        print(f"Type: {analysis.query_type.value}")
        print(f"Strategy: {analysis.retrieval_strategy.value}")
        print(f"Complexity: {analysis.complexity_score}")
        print(f"Reasoning: {analysis.requires_reasoning}")
        print(f"Multi-hop: {analysis.requires_multi_step}")
        print(f"Confidence: {analysis.confidence}")
        print(f"Why: {analysis.reasoning}") 
        if analysis.sub_questions:
            print(f"Sub-questions:")
            for sq in analysis.sub_questions:
                print(f" -> {sq}")    
      