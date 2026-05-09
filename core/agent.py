"""
This is the Orchestrator. This is the most important file in the whole project.
It ties all 6 steps together into one intelligent loop.

Think of it like an orchestrator. Ecah component we built is a musician, the agent tells each one wheh to play and coordinates them into a coherent performance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from query_rewriter import QueryReWriter
from query_analyser import QueryAnalyser
from retriever import HybridRetriever, RetrievedChunk
from reranker import Reranker
from evaluator import ContextEvaluator
from generator import Generator, GeneratedAnswer

@dataclass
class AgentTrace:
    """
    Complete reasoning trace of the agent's decision process.
    This is what gets displayed in the UI - full transparency
    """

    query: str
    timestamp: str
    
    # Step results
    query_type: str = " "
    complexity_score: float = 0.0
    rewrite_applied: bool = False
    original_query: str = ""
    rewritten_query: str = ""
    retrieval_strategy: str = ""
    chunks_retrieved: int = 0
    chunks_after_reranking: int = 0
    evaluation_attempts : int = 0
    evaluation_passed: bool = False
    generation_mode: str = ""
    tokens_used: int = 0

    # Final output
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)



    # Performance
    total_time_seconds: float = 0.0
    success: bool = False
    error: str = ""

    #This is the reasoning trace that makes our system transparent.
    # Every decision the agent makes gets recorded here

class AgenticRAG:
    """
    The main Agentic RAG orchestrator.

    Coordibates all components in the correct sequence
    1. Analyse - understand the question
    2. Rewrite - improve the question
    3. Retrieve - find relevant chunks
    4. Rerank - filter best chunks
    5. Evaluate - enough context? if not retry
    6. Generate - produce answer with reasoning

    Maintains a full reasoning trace for transparency
    """  

    MAX_RETRIES = 3

    def __init__(self):
        print("Intialising Agentic RAG...")
        self.analyser = QueryAnalyser()
        self.rewriter = QueryReWriter()
        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.evaluator = ContextEvaluator()
        self.generator = Generator()

        print("All components ready!")
        print("Agent is ready to answer questions!")

    def ask(self, query: str) -> AgentTrace:
        """
        Main entry point - ask the agent a question.
        Returns complete Agent Trace with answer and reasoning
        """ 

        start_time = datetime.now()

        trace = AgentTrace(
            query=query,
            timestamp=start_time.strftime("%Y-%m-%d %H:%M:%S"),
            original_query=query
        )  
        print(f"Agent Processing: {query[:60]}....")

        try:
            print("STEP 1: Query Analysis")
            analysis = self.analyser.analyse(query)
            trace.query_type = analysis.query_type.value
            trace.complexity_score = analysis.complexity_score
            trace.sub_questions = analysis.sub_questions

            print(f"Type: {analysis.query_type.value} |"
                  f"Complexity: {analysis.complexity_score}"
                  f"Confidence: {analysis.confidence}")
            
            print("Step 2: Query Rewritting")
            rewrite_result = self.rewriter.rewrite(query, analysis)

            trace.rewrite_applied = self.rewriter.rewrite(query, analysis)

            trace.rewrite_applied = rewrite_result.was_written
            trace.rewritten_query = rewrite_result.rewritten

            #Use rewritten query for retrieval
            search_query = rewrite_result.rewritten

            #Next steps, retrieve, rerank and evaluate loop.
            final_chunks =  []
            evaluation_result = None

            for attempt in range(1, self.MAX_RETRIES + 1):
                print(f"Retrieval Attempt {attempt}/ {self.MAX_RETRIES}") 

                #Determine retrieval k based on query type
                k = 8 if analysis.query_type.value == 'multi-hop' else 5

                chunks, strategy = self.retriever.search(
                    search_query, k=k
                )
                trace.retrieval_strategy = strategy
                trace.chunks_retrieved = len(chunks)

                print(f"Retrieved {len(chunks)} chunks via {strategy} search")

                print("Reranking")
                final_chunks, scoring_details = self.reranker.process(
                    search_query, chunks, top_k=3
                )
                trace.chunks_after_reranking = len(final_chunks)

                print("Context Evaluation")
                evaluation_result = self.evaluator.evaluate(
                    query, final_chunks, attempt_number=attempt
                )
                trace.evaluation_attempts = attempt

                if evaluation_result.is_sufficient:
                    trace.evaluation_passed = True
                    print(f"Context sufficient - proceeding")
                    break
                else:
                    print(f"Context insufficient - {evaluation_result.reasoning1}")

                    if attempt < self.MAX_RETRIES:
                        #Expand the search query for next attempt
                        search_query = (
                            f"{search_query}"
                            f"Kenya financial inclusion statistics data"
                        )
                        print(f" Expanding query for retry...")

                    print("Answer Generation")
                    generated = self.generator.generate(
                        query=query,
                        chunks=final_chunks,
                        analysis=analysis,
                        sub_questions=rewrite_result.sub_questions
                    )    

                    # Build complete trace
                    trace.answer = generated.answer
                    trace.citations = generated.citations
                    trace.reasoning_steps = generated.reasoning_steps
                    trace.generation_mode = generated.generation_mode
                    trace.tokens_used = generated.tokens_used
                    trace.success = True

                    elapsed = (datetime.now() - start_time).total_seconds()
                    trace.total_time_seconds = round(elapsed, 2)

                    print(f"Agent Complete in {elapsed:.1f}s")

        except Exception as e:
            trace.error = str(e)
            trace.answer = f"Agent encountered an error: {str(e)}"
            trace.success = False
            print(f"Agent error: {str(e)}")
        return trace    
    def print_trace(self, trace: AgentTrace):
        """Print a formatted reasoning trace to terminal."""
        print(f"\n{'━'*55}")
        print(f"REASONING TRACE")
        print(f"{'━'*55}")
        print(f"Query:        {trace.query[:60]}")
        print(f"Timestamp:    {trace.timestamp}")
        print(f"")
        print(f"Step 1 — Analysis:")
        print(f"  Type:       {trace.query_type}")
        print(f"  Complexity: {trace.complexity_score}")
        print(f"")
        print(f"Step 2 — Rewriting:")
        print(f"  Applied:    {trace.rewrite_applied}")
        print(f"  Result:     {trace.rewritten_query[:50]}...")
        print(f"")
        print(f"Step 3 — Retrieval:")
        print(f"  Strategy:   {trace.retrieval_strategy}")
        print(f"  Retrieved:  {trace.chunks_retrieved} chunks")
        print(f"")
        print(f"Step 4 — Reranking:")
        print(f"  Kept:       {trace.chunks_after_reranking} chunks")
        print(f"")
        print(f"Step 5 — Evaluation:")
        print(f"  Attempts:   {trace.evaluation_attempts}")
        print(f"  Passed:     {trace.evaluation_passed}")
        print(f"")
        print(f"Step 6 — Generation:")
        print(f"  Mode:       {trace.generation_mode}")
        print(f"  Tokens:     {trace.tokens_used}")
        print(f"")
        print(f"Result:")
        print(f"  Success:    {trace.success}")
        print(f"  Time:       {trace.total_time_seconds}s")
        print(f"{'━'*55}")
        print(f"\n ANSWER:\n{trace.answer}")
        print(f"\n CITATIONS:")
        for c in trace.citations:
            print(f"  {c}")

if __name__ == "__main__":
    print("Testing Agentic RAG system")
    
    agent = AgenticRAG()

    test_queries = [
        "What is Kenya's financial inclusion rate?",
        "Why is Turkana's inclusion rate lower than Nairobi and what should be done?",
        "Compare the gender gap in 2006 vs 2024 and explain what caused the change",
    ]
    for query in test_queries:
        trace = agent.ask(query)
        agent.print_trace(trace)
        

