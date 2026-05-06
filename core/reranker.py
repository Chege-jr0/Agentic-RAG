"""
Retriever = Librarian finding 8 books that might help
Reranker = You reading the first page of each book and deciding which 3 are actually useful
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util
from retriever import RetrievedChunk

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class Reranker:
    """
    Reranks retrieved chunks by computing semantic similarity between the 
    query and each chunk more carefully than the initial retrieval step.

    Use cross-attention style scoring - compares query and chunk together 
    rather than seperately
    """

    def __init__(self):
        print("Loading reranker...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("Reranker ready!")

    def score_chunk(self, query: str, chunk: RetrievedChunk) -> float:
        """
        Score a single chunk against the query.
        Returns similarity score between 0.0 and 1.0

        """   
        query_embedding = self.model.encode(
            query, convert_to_tensor=True
        ) 

        chunk_embedding = self.model.encode(
            chunk.text, convert_to_tensor=True
        )

        #Cosine similarity between query and chunk
        #util.cos_sim() from sentence_transformers is more precise
        similarity = util.cos_sim(
            query_embedding, chunk_embedding
        ).item()

        return round(float(similarity), 3)
    
    def rerank(
        self, 
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 3,
        threshold: float = 0.3
    ) -> tuple[list[RetrievedChunk], list[dict]]:
        """
        Rerank a list of chunks by relevance to the query.

        threshold: minimum score to keep a chunk
                   chunks below this are filtered out

        top_k: maximum chunks to return after reranking

        Returns reranked chunks AND scoring_details for 
        the reasoning trace.            
        """
        if not chunks:
            return [], []
        
        print(f"Reranking {len(chunks)} chunks....")

        scoring_details = []

        for chunk in chunks:
            new_score = self.score_chunk(query, chunk)

            scoring_details.append({
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "original_score": chunk.score,
                "rerank_score": new_score,
                "text_preview": chunk.text[:60] + "...",
                "kept": new_score >= threshold
            })

            #Update chunk score with reranker score
            chunk.score = new_score

        #Filter by threshold
        filtered = [
            c for c in chunks if c.score >= threshold
        ]    

        #Sort by score descending
        filtered.sort(key=lambda x: x.score, reverse=True)

        # Return top k
        final = filtered[:top_k]

        print(f"Kept {len(final)} chunks after reranking")

        print(f"Score range: "
              f"{min(c.score for c in final):.3f} - "
              f"{max(c.score for c in final):.3f}"
              if final else "No chunks dashboard")
        
        return final, scoring_details
    
    def deduplicate(
            self, 
            chunks: list[RetrievedChunk],
            similarity_threshold: float = 0.92

    ) -> list[RetrievedChunk]:
        """
        Remove near duplicate chunks
        If two chunks are very similar keep only the higher scored one.
        Prevents the same information appearing twice in context
        """

        if len(chunks) <= 1:
               return chunks
        kept = []
        embeddings = self.model.encode(
            [c.text for c in chunks],
            convert_to_tensor = True
        )  

        for i, chunk in enumerate(chunks):
            is_duplicate = False

            for j, kept_chunk in enumerate(kept):
                similarity = util.cos_sim(
                    embeddings[i],
                    embeddings[chunks.index(kept_chunk)]
                ).item()

                if similarity > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(chunk)  

        if len(kept) < len(chunks):
            print(f"Removed {len(chunks) - len(kept)} duplicate_chunks") 

        return kept    

    def process(
            self, 
            query: str,
            chunks: list[RetrievedChunk],
            top_k: int = 3,
            threshold: float = 0.3
    ) -> tuple[list[RetrievedChunk], list[dict]]:
        """
        Full reranking pipeline:
        1. Rerank by relevance score
        2. Deduplicate similar chunks
        3. Return top k with scoring details
        """ 
        #Step 1 - Rerank
        reranked, details = self.rerank(query, chunks, top_k, threshold)

        #Step 2 - Deduplicate
        final = self.deduplicate(reranked)  

        return final, details   

if __name__ == "__main__":
    from core.retriever import HybridRetriever

    print("Testing Reranker...")

    retriever = HybridRetriever()
    reranker = Reranker()

    query = "What are the barriers to financial inclusion in Kenya?" 
    print(f"Query: {query}")

    #Step 1 - Retrieve
    chunks, strategy = retriever.search(query, k=5)
    print(f"Retrieved {len(chunks)} chunks via {strategy} search")

    #Step 2 - Rerank
    final_chunks, details = reranker.process(query, chunks, top_k=3) 

    print(f"Scoring Details:")
    for d in details:
        status = "kept" if   d["kept"] else "removed"
        print(f"{status} | original = {d['text_preview']}"
              f"-> rerank={d['rerank_score']} | {d['text_preview']}")  

    print(f"Final Chunks after reranking: {len(final_chunks)}")
    for i, chunk in enumerate(final_chunks):
        print(f"Chunk {i+i}: score={chunk.score} | {chunk.text[:80]}...")       
