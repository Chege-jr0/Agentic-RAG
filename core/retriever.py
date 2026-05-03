"""
Remember the first RAG we created vectorstore.as_retriever(search_kwargs={"k": 5}) thats it one line, one strategy, always vector search
For Agentic RAG retriver: it can use 3 strategies, the agent picks the right one:
1. Vector Search => find semantically similar chunks
2. Keyword Search => find exact term matches
3. Hybrid => combines both with weighted scoring
"""

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-l6-v2"

@dataclass
class RetrievedChunk:
    """
    A python decorator that automatically creates a clean class
    insteed of writing __init__, __repr__ manually
    Every chunk we retrieve will be wrapped in this source, score and method all tracked
    Represents a single retrieved chunk with metadta.
    dataclass automatically creates __init__ and __repr__
    """
    text: str
    source: str
    chunk_index: str
    score: float
    retrieval_method: str


class HybridRetriever:
    """
    Hybrid retriever combining vector search and keyword search.
    Agent selects the best strategy based on query type
    We used classes instead of functions because the embedding model takes 2 seconds to load
    A class loads it once when intialized
    Every subsequent search reuses the same loaded model
    Much faster than reloading the model on evry search
    """   
    def __init__(self):
        print("Loading retriever...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path = CHROMA_PATH)
        self.collection = self.client.get_collection("agentic_rag")
        print("Retriever ready!")

    def vector_search(
            self, query: str, k: int = 5
    )  -> list[RetrievedChunk]:
        """
        Semantic search - finds chunks similar in MEANING.
        Best for: conceptual questions, paraphrased queries.
        """  

        # Embed the query using same model as ingestion
        # Converts the question into a vector using same model we used during ingestion
        # This is critical because if we used different models, it would create incompatible vector spaces
        query_embedding = self.model.encode([query]).tolist()[0]

        results = self.collection.query(
            query_embeddings = [query_embedding],
            n_results = min(k, self.collection.count()),
            include = ["documents", "metadatas", "distances"]
        )

        chunks = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            # Convert distance to similarity score(distance to similarity conversion)
            # ChromaDB returns cosine distance (0=identical, 2=opposite)
            # We convert to similarity (1=identical, 0=opposite)
            similarity = round(1 - (dist /2 ), 3)
            chunks.append(RetrievedChunk(
                text = doc,
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index"),
                score = similarity,
                retrieval_method= "vector"
            ))
        return chunks
    
    def keyword_search(
            self, query: str, k: int =5
    ) -> list[RetrievedChunk]:
        """
        Exact keyword matching - finds chunks containing the actual words from the query.
        Best for: specific names, numbers, dates, counties
        """
        # Get all documents from ChromaDB
        all_docs = self.collection.get(
            include = ["documents", "metadatas"]
        )

        query_words = set(query.lower().split())

        #Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "of", "and",
            "or", "but", "with", "what", "how", "why",
            "which", "who", "where", "when", "does", "do"
        }

        query_words = query_words - stop_words

        scored_docs = []

        for doc, meta in zip(
            all_docs["documents"],
            all_docs["metadatas"]
        ):
            doc_words = set(doc.lower().split())

            # Count mathing words
            matches = len(query_words & doc_words)

            if matches > 0:
                #Score = matched words / total_query words
                score = round(matches / len(query_words), 3)
                scored_docs.append((doc, meta, score))

          # Count matching words
        scored_docs.sort(key=lambda x: x[2], reverse=True)  

        chunks = []
        for doc, meta, score in scored_docs[:k]:
            chunks.append(RetrievedChunk(
                text=doc,
                source= meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", 0),
                score = score,
                retrieval_method="keyword"
            ))
        return chunks  


    def hybrid_search(
            self, 
            query: str,
            k: int = 5,
            vector_weight: float = 0.7,
            keyword_weight: float = 0.3
    )  -> list[RetrievedChunk]:
        """
        Combines vector search and keyword search with weighted scoring.
        vector_weight = 0.7 means semantic search counts more.
        keyword_weight = 0.3 means exact matching is secondary.
        Best for: most queries - balances meaning and precision.
        """
        vector_results = self.vector_search(query, k = k*2)
        keyword_results = self.keyword_search(query, k = k*2)

        #Build combined score dictionaru
        combined = {}

        for chunk in vector_results:
            key = chunk.text[:100]
            combined[key] = {
                "chunk": chunk,
                "vector_score": chunk.score,
                "keyword_score": 0.0
            }

        for chunk in keyword_results:
            key = chunk.text[:100]
            if key in combined:
                combined[key]["keyword_score"] = chunk.score

            else:
                combined[key] = {
                    "chunk": chunk,
                    "vector_score": 0.0,
                    "keyword_score": chunk.score
                } 

        # Calculate weighted hybrid score
        results = []
        for key, data in combined.items():
            hybrid_score = round(
                (data["vector_score"] * vector_weight) +
                (data["keyword_score"] * keyword_weight), 3
            )   
            chunk = data["chunk"]
            chunk.score = hybrid_score
            chunk.retrieval_method = "hybrid"
            results.append(chunk)

        # Sort by hybrid score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]    
    

    def search(
            self, 
            query: str,
            strategy: str = "auto",
            k: int = 5
    ) -> tuple[list[RetrievedChunk], str]:
        """
        Main Search Function - agent calls this.
        Strategy: 'auto' / 'vector' / 'keyword' / 'hybrid'

        'auto' lets the retriever decide based on query characteristics.
        Returns chunks AND the strategy used.
        """
        if strategy == "auto":
            strategy = self.select_strategy(query)

        print(f"Using {strategy} search strategy")

        if strategy == "vector":
            chunks = self.vector_search(query, k)

        elif strategy == "keyword":
            chunks = self.keyword_search(query, k)

        else:
            chunks = self.hybrid_search(query, k)

        return chunks, strategy
    def select_strategy(self, query: str) -> str:
        """
        Automatically select retrieval strategy based on query.
        This is a simple heuristic - the agent can override this.
        """   

        query_lower = query.lower()
        words = query_lower.split()

        # Signals that keyword search works better 
        # Specific county names, numbers, product names
        specific_terms = [
            "turkana", "nairobi", "mombasa", "kisumu",
            "nakuru", "kiambu", "percent", "%",
            "m-pesa", "mpesa", "fuliza", "shwari",
            "hustler", "sacco", "exactly", "specific",
            "2024", "2023", "2022", "2021", "2020"
        ]  

        has_specific_terms = any(
            term in query_lower for term in specific_terms
        )   

        #Short queries with specific terms -> keyword
        if len(words) <= 4 and has_specific_terms:
            return "keyword"   

        #Long conceptual queries => vector
        if len(words) > 10 and not has_specific_terms:
            return "vector" 

        #Everything else => hybrid
        return "hybrid"   


if __name__ == "__main__":
    print("Testing HybridRetriever...")

    retriever = HybridRetriever()

    test_queries = [
        "What is the financial inclusion in Kenya?",
        "Turkana County exclusion rate",
        "Why are women exclude from financial services?"
    ]

    for query in test_queries:
        print(f"Query: {query}")
        chunks, strategy = retriever.search(query, k=3)
        print(f"Strategy used: {strategy}")
        print(f"Chunks found{len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: score={chunk.score} | source={chunk.source}")
            print(f"Text Preview: {chunk.text[:80]}...")



