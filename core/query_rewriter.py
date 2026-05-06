"""
What is Query Rewriting?
This is one of the most powerful ideas in Agentic RAG. Instead of searching with the user's exact words,
the agent transforms the query to get better results

1. EXPANSION - add contexy to vague queries
"how bad is turkana?"
"]"Turkana county financial inclusion exclusion rate barriers statistics Kenya 2024"

2. CLARIFICATION - fix ambigous queries
"what is the age gap?"
"What is the gender gap in financial inclusion Kenya"

3. DECOMPOSITION - break complex queries into parts
"compare counties and explain why and recommend product"
-> Sub_question 1: "which counties have lowest inclusion?"
   Sub_question 2: "what causes low_inclusion in those counties?"
   Sub_question 3: ""what financial products suit excluded Kenyans"
"""

import ollama
from dataclasses import dataclass
from query_analyser import QueryAnalysis, QueryType

@dataclass
class RewrittenQuery:
    """
    Result of query rewritting.
    Contains original, rewritten and any sub_questions
    """
    original: str
    rewritten: str
    sub_questions: list[str]
    rewrite_type: str  #expansion / clarification / decomposition
    was_written: bool  # False if original was good enough


class QueryReWriter:
    """
    Rewrites user queries to improve retrieval quality.

    Uses two approaches"
    1. Rule-based - fast, handles common patterns, saves tokens, more powerful
    2. LLM-based - for complex rewrites that need understanding
    """ 
    DOMAIN_EXPANSIONS = {
        "mpesa": "M-Pesa mobile money Kenya",
        "m-pesa": "M-Pesa mobile money Kenya",
        "inclusion": "financial inclusion access banking Kenya",
        "exclusion": "financial exclusion barriers Kenya",
        "gap": "gap difference percentage points",
        "counties": "Kenya counties regions financial access",
        "barriers": "barriers obstacles challenges financial inclusion",
        "sacco": "SACCO savings credit cooperative Kenya",
        "fuliza": "Fuliza M-Pesa overdraft Safaricom Kenya",
        "hustler": "Hustler Fund government credit Kenya",
    }   

    #Vague words that need expansion
    VAGUE_WORDS = {
        "bad", "good", "high", "low", "poor", "great",
        "struggling", "doing well", "better", "worse",
        "how", "what about", "tell me about"
    }

    def rule_based_rewrite(self, query: str) -> str:
        """
        Fast rule-based query expansions.
        Handles common patters without needing an LLM
        """

        query_lower = query.lower()
        words = query_lower.split()
        rewritten = query

        # Expand domain-specific terms
        for term, expansion in self.DOMAIN_EXPANSIONS.items():
            if term in query_lower and expansion not in query_lower:
                rewritten= rewritten.replace(term, expansion)

        # If query is very short - add context  
        if len(words) <= 4:
            rewritten = f"{rewritten} Kenya financial inclusion statistics data"

        # If query has vague words - add specificity
        has_vague = any(word in words for word in self.VAGUE_WORDS)
        if has_vague:
            rewritten = f"{rewritten} statistics percentage rata data Kenya" 

        return rewritten.strip()

    def needs_llm_rewrite(self, query: str, analysis: QueryAnalysis) -> bool:
        """
        Decide if we need LLM rewritinng or rule-based is enough.
        LLM rewriting is slower so we only use it when necessary
        """     

        # Multi-hop queries benefit most from LLM rewritting
        if analysis.query_type == QueryType.MULTI_HOP:
            return True

        #Very vague short queries need LLM understanding
        words = query.split()
        if len(words) <= 3:
            return True

        #Queries with pronouns are ambigous
        pronouns = {"it", "they", "them", "that", "these"}
        if any(p in query.lower().split() for p in pronouns):
            return True 

        return False 
    
    def llm_rewrite(self, query: str, analysis: QueryAnalysis) -> str:
        """
        Use TinyLlama to rewrite complex or ambigous queries
        Returns a better searh query.
        """
        prompt = f"""
        You are a search query optimizer for a Kenya Financial Inclusion database.
        
        Rewrite the following query to be more specific and searchable.
        The database contains information about:
        - M-Pesa adoption and subscriber data
        - Financial inclusion rates by county
        - Gender gaps in financial access
        - Barriers to financial inclusion
        - Financial products (M-Shwari, Fuliza, SACCOs, Hustler Fund)
        - Urban vs rural access differences

        Original query: {query}

  
        Rules for rewriting:
        1. Keep it under 20 words
        2. Include specific Kenya financial inclusion terms
        3. Remove vague words like "good" or "bad"
        4. Add relevant context words
        5. Return ONLY the rewritten query — no explanation\
        
        Rewritten query:

"""
        try: 
            response = ollama.chat(
                model= "tinyllama",
                messages=[{"role": "user", "content": prompt}]
            )
            rewritten = response["messages"]["content"].strip()

            # Clean up - remove quotes if model added them
            rewritten = rewritten.strip('"\'')

            # Sanity Check - if rewrite is too long or weird use original
            if len(rewritten.split()) > 25 or len(rewritten) < 5:
                return query

        except Exception as e:
            return query  
         
   # LLM decomposition
    def llm_decompose(self, query: str) -> list[str]:
        """
        Use TinyLlama to decompose a complex multi-hop query into simpler questions
        """

        prompt = f"""
        Break this complex question into 2-3 simple sub-questions.
        Each sub-question should be answerable indepedently.

        Complex question: {query}

        Return ONLY the sub-questions, one per line, numbered:
        1. 
        2.
        3. 
      """
        try: 
            response = ollama.chat(
                model = "tinyllama",
                messages=[{"role": "user", "content": prompt}]

            )
            content = response["message"]["content"].strip()

            # Parse numbered list
            lines = content.split("\n")
            sub_questions = []

            for line in lines:
                line = line.strip()
                #Remove numbering like "1", "2"
                if line and line[0].isdigit() and "." in line:
                    question = line.split(".", 1)[1].strip()
                    if len(question) > 10:
                        sub_questions.append(question)

                #Fallback if parsing fails
                if not sub_questions:
                    return[query]
                
                return sub_questions[:3] # Max sub-questions


        except Exception as e:
            return [query]
        
    def rewrite(self, query: str, analysis: QueryAnalysis) -> RewrittenQuery:
        """
        Main rewritting function - called by the agent.
        Decides which rewriting approach to use and applies it.
        """ 
        sub_questions = []
        rewrite_type = "none"
        was_rewritten = False

        # Multi-hop -> decompose into sub-questions
        if analysis.query_type == QueryType.MULTI_HOP:
            print("Decomposing multi-hop query....")

            if self.needs_llm_rewrite(query, analysis):
                sub_questions = self.llm_decompose(query)

            else:
                sub_questions = analysis.sub_questions or [query]  

            rewritten = sub_questions[0] if sub_questions else query
            rewrite_type = "decomposition"
            was_rewritten = True  
         
         # LLM rewrite for complex/ambigous queries
        elif self.needs_llm_rewrite(query, analysis):
            print("Rewriting query with LLM...")
            rewritten = self.llm_rewrite(query, analysis)
            rewrite_type = "llm_expansion"
            was_rewritten = rewritten != query

        # Rule-based rewrite for simple expansion
        else:
            print("Applying rule-based rewrite...")  
            rewritten = self.rule_based_rewrite(query)
            rewrite_type = "rule_expansion"
            was_rewritten = rewritten != query

        if was_rewritten:
            print(f"Original: {query[:50]}...")
            print(f"Rewritten: {rewritten[:50]}...")

        else:
            print("Query is clear - no rewrite needed")   

        return RewrittenQuery(
            original = query,
            rewritten = rewritten,
            sub_questions = sub_questions,
            rewrite_type = rewrite_type,
            was_written = was_rewritten
        ) 

if __name__ == "__main__":
    from query_analyser import QueryAnalyser

    print("Testing QueryWriter...")

    analyser = QueryAnalyser()
    rewriter = QueryReWriter()

    test_queries = [
        "what is mpesa?",
        "how bad is turkana?",
        "Compare the gender gap and explain why it changed and what should policy focus on next",
        "What are the barriers to financial inclusion in rural Kenya?",
        "tell me about it"
    ]  

    for query in test_queries:
        print(f"Original: {query}")
        analysis = analyser.analyse(query)
        result = rewriter.rewrite(query, analysis)
        print(f"Type: {result.rewrite_type}")
        print(f"Rewritten: {result.rewritten}")

        if result.sub_questions:
            print(f" Sub-questions:")
            for sq in result.sub_questions:
                print(f" -> {sq}")  



