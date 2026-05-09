"""
This is the reasoning Engine.
This is what makes this different from our basic RAG genèrator

In our RAG previous RAG, we had one shot, no reasoning, no citations, no structure

In this RAG, the response will be 3 modess based on query complexity
1. Direct - simple factual answer with citation
2. Structured - complex answer with sections.
3. Multi-step - step by step reasoning chain.
"""

import ollama
from dataclasses import dataclass
from retriever import RetrievedChunk
from query_analyser import QueryAnalysis, QueryType

@dataclass
class GeneratedAnswer:
    """
    Complete answer with reasoning trace.
    Contains everything shown in the UI
    """
    answer: str
    citations: list[str]
    reasoning_steps: list[str]
    generation_mode: str
    confidence: float
    tokens_used: int

class Generator:
    """
    Generated answers from retrieved context.
    Uses different strategies based on query complexity.

    Three modes:
    1. direct - simple factual questions
    2. structured - complex questions needing organisation
    3. multi-step - multi-hop questions needing reasoning chain.
    """   

    MODEL = "tinyllama"

    #Context window management
    MAX_CONTEXT_WORDS = 600 # Keep well within TinyLlama's limit
    MAX_CHUNK_WORDS = 150 # Max words per chunk in context

    def __init__(self):
        print("Generator ready..")

    def build_context(self, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
        """
        Build context string from chunks.
        Truncates long chunks to stay within token limit.
        Returns context string and list of citations
        """    
        context_parts = []
        citations = []
        total_words = 0

        for i, chunk in enumerate(chunks):
            words = chunk.text.split()

            #Truncate if too long
            if len(words) > self.MAX_CHUNK_WORDS:
                words = words[:self.MAX_CHUNK_WORDS]
                text = " ".join(words) + "..."
            else:
                text = chunk.text

            # Check total context size
            if total_words + len(words) > self.MAX_CONTEXT_WORDS:
                break

            context_parts.append(
                f"[Source {i+1}] {text}"
            ) 
            citations.append(
                f"Source {i+1}: {chunk.source}"
                f"(chunk {chunk.chunk_index}), "
                f"relevance: {chunk.score:.2f}"
            )
            total_words += len(words)

        context = "\n\n".join(context_parts)
        return context, citations    
    
    # Direct Response
    def generate_direct(self, query: str, context: str, analysis: QueryAnalysis) -> tuple[str, list[str]]:
        """
        Mode 1 - Direct answer for simple factual questions.
        Short, precise, to the point
        """
        prompt = f"""
        You are a Kenya Financial expert.
        Answer the question using only the context provided
        Be concise 2-3 sentences maximum
        If the context doesn't contain the answe say so clearly

        Context: {context}

        Question: {query}

        Answer:
        """
        reasoning_steps = [
            "Query classified as simple factual question",
            "Using direct generation mode",
            f"Context: {len(context.split())} words from {len(context.split('[Source]'))-1} sources"
        ]

        try:
            response = ollama.chat(
                model = self.MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response["message"]["content"].strip()
            return answer, reasoning_steps
        except Exception as e:
            return f"Generation failed: {str(e)}", reasoning_steps
        

    # Strucutured Response    
    def generate_structured(self, query: str, context: str, analysis: QueryAnalysis) -> tuple[str, list[str]]:
        """
        Mode 2 - Strucutured answer for complex questions.
        Organises information into clear sections.
        """    
        # Build additional instructions based on analysis
        extra_instructions = ""
        if analysis.requires_comparison:
            extra_instructions += "\n - Compare the items clearly using specific numbers"
        if analysis.requires_reasoning:
            extra_instructions += "\n - Explain the reasons behind the patterns"

        prompt = f"""
        You are a Kenya Financial inclusion expert providing a detailed analysis.
        Use ONLY the context provided. Structure your answer clearly.
        {extra_instructions}

        Context: {context}

        Questions : {query}

        Provide a structurd answer with:
        1. Direct answer to the question
        2. Key supporting evidence ( with specific numbers)
        3. Brief explanation of implications

        Answer:
        """    
        reasoning_steps = [
            "Query classified as complex question",
            "Using structured generation mode",
            f"Analysis flags: comparison = {analysis.requires_reasoning},"
            f"Context: {len(context.split())} words provided"
        ]  

        try:
            respose = ollama.chat(
                model = self.MODEL,
                messages=[{"role": "user", "content": prompt}]

            )
            answer = respose["message"]["content"].strip()
            return answer, reasoning_steps
        except Exception as e:
            return f"Generation failed: {str(e)}", reasoning_steps
        

        # Multi-step Generation
    def generate_multi_step(self, query: str, context: str, analysis: QueryAnalysis, sub_questions: list[str]) -> tuple[str, list[str]]:
        """
        Mode 3 - Multi-step reasoning for complex multi-hop questions.
        Answers each sub-question then synthesises into final answer
        """  
        reasoning_steps = [
            "Query classifies as multi-hop question",
            f"Decomposed into {len(sub_questions)} sub-questions",
        ]  

        #Answer Each sub_question
        sub_answers = []
        for i, sub_q in enumerate(sub_questions[:3]):
            reasoning_steps.append(f"Step {i+1}: Answering - {sub_q[:50]}...")

            sub_prompt = f"""
        Using the context below, answer this specific quesion briefly.
        """
            try:
                response = ollama.chat(
                    model = self.MODEL,
                    messages=[{"role": "user", "content": sub_prompt}]
                )
                sub_answer = response["mesage"]["content"].strip()
                sub_answers.append(f"{sub_q}\n {sub_answer}")
            except Exception as e:
                sub_answer.append(f"{sub_q} \n Unable to answer this part")

        # Synthesis sub-answers into final answer
        reasoning_steps.append("Synthesising sub-answers into final response")

        synthesis_prompt = f"""
        You have answered several sub-questions about Kenya financial inclusion.
        Synthesise these answers into one coherent final answer.

        Sub-question answers:
        {chr(10).join(sub_answer)}
       
        Original question: {query}
        
        Provide a clear synthesised answer:
        """    

        try:
            response = ollama.chat(
                model = self.MODEL,
                messages= [{"role": "user", "content": synthesis_prompt}]
            )
            final_answer = response["message"]["content"].strip() 

            # Include sub-answers in output
            full_answer = final_answer + "\n\n---\n Reasoning Steps:\n"
            for sub_a in sub_answers:
                full_answer += f"\n {sub_a}"

            return full_answer, reasoning_steps    

        except  Exception as e:    
            return "\n\n".join(sub_answers), reasoning_steps

    # Main Generate Function and Test
    def generate(self, query: str, chunks: list[RetrievedChunk], analysis: QueryAnalysis, sub_questions: list[str] = None) -> GeneratedAnswer:
        """
        Main generation function - called by the agent.
        Selects the right mode and generates the answe
        """  
        print(f" Generating answer...")

        # Buil context from chunks
        context, citations = self.build_context(chunks)

        # Select generation mode
        if analysis.query_type == QueryType.MULTI_HOP and sub_questions:
            mode = "multi_step"
            print(f"Mode: multi-step reasoning ({len(sub_questions)}) steps")
            answer, reasoning_steps = self.generate_multi_step(
                query, context, analysis, sub_questions
            )

        elif analysis.query_type.value in ["complex"] or \
             analysis.requires_comparison or \
             analysis.requires_reasoning:
            mode = "structured"
            print(f"Mode: strucutured generation")
            answer, reasoning_steps = self.generate_structured(
                query, context, analysis
            )

        else: 
            mode = "direct"
            print(f"Mode: direct generation")
            answer, reasoning_steps = self.generate_direct(
                query, context, analysis
            )

        # Estimate tokens used
        tokens_used = len((context + query + answer).split()) * 1.3

        print(f"Answer generated! (~{int(tokens_used)} tokens used)")  

        return GeneratedAnswer(
            answer=answer,
            citations=citations,
            reasoning_steps=reasoning_steps,
            generation_mode=mode,
            confidence=0.8 if chunks else 0.3,
            tokens_used=int(tokens_used)
        )      

if __name__ == "__main__":
    from retriever import HybridRetriever
    from reranker import Reranker
    from query_analyser import QueryAnalyser

    print("Testing Generator...")

    retriever = HybridRetriever()
    reranker = Reranker()
    analyser = QueryAnalyser()
    generator = Generator()

    test_queries = [
        "What is Kenya's financial inclusion rate?"
        "Why is Turkana's inclusion rate lower than Nairobi?"

    ]

    for query in test_queries:
        print(f"\n Query: {query}")

        analysis = analyser.analyse(query)
        chunks, _ = retriever.search(query, k=5)
        final_chunks, _ = reranker.process(query, chunks)

        result = generator.generate(
            query, final_chunks, analysis
        )

        print(f"\n Answer:")
        print(result.answer)
        print(f"\n Citations:")
        for c in result.citations:
            print(f" {c}")

        print(f"\n Tokens used: {result.tokens_used}")
        print(f"Mode: {result.generation_mode}")    
