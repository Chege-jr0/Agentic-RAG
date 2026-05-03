import chromadb
import os
import json
from sentence_transformers import SentenceTransformer
from chromadb import Settings

# Paths
CHROMA_PATH = "chroma_db"
DOCUMENTS_PATH = "documents"

#Embedding model from sentence transformers
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

print("Ingest Module Loaded")

"""
This file takes in any document you give it, splits them into chunks, embeds each chunk using sentence-transformers and stores everything in ChromaDB
"""

def get_chroma_client():
    "Intialiaze ChromaDB client with persistent storage."
    client = chromadb.PersistentClient(path = CHROMA_PATH)
    return client

def get_or_create_collection(client, collection_name="agentic_rag"):
    """
    Get existing collection or create a new one.
    Retrieves an existing collection from your vector database or creates it if doesn't exist
    "cosine" refers to cosine similarity meaning vectors are compared based on angle not magnitude
    hnsw:space means Hierarchical Navigable Small World(fast vector search algorithm)
    """
    collection = client.get_or_create_collection(
        name = collection_name,
        metadata = {"hnsw:space": "cosine"}
    )
    return collection


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    Overlap ensures context is not lost at chunk boundaries
    overlap=50 means each chunk shares 50 words with the next
    This prevents loosing context at chunk boundaries.
    """
    chunks = []
    words = text.split()

    if len(words) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start += chunk_size - overlap

    return chunks    

def load_documents() -> list[dict]:
    """
    Load all documents from documents folder.
    Support .txt and .md files
    Returns list of dicts with text and metadata,.
    metadata means the filename and filepath 
    """
    os.makedirs(DOCUMENTS_PATH, exist_ok=True)
    documents = []

    for filename in os.listdir(DOCUMENTS_PATH):
        filepath = os.path.join(DOCUMENTS_PATH, filename)

        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()

                if text:
                    documents.append({
                        "text": text,
                        "filename": filename,
                        "filepath": filepath,
                        "type": "text"
                    })
                    print(f"Loaded: {filename} ({len(text)} chars)")

    print(f"Total documents loaded: {len(documents)}")
    return documents 


def ingest_documents():
    """
    Main Ingestion pipeline Load documents, Chunk each document, embed chunks and stores in ChromaDB
    """
    print(f"Starting document ingestion. Loading {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading Documents...")
    documents = load_documents()

    if not documents:
        print("No documents found!")

        print(f"Add .txt or .md files to the {DOCUMENTS_PATH} folder")
        return
    
    # Intialize ChromaDB
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    #Process each document
    all_chunks = []
    all_embeddings = []
    all_ids = []
    all_metadata = []

    chunk_counter = 0

    for doc in documents:
        print(f"Processing {doc['filename']}")

        chunks = chunk_text(doc["text"])
        print(f"Split into {len(chunks)} chunks")

        #Embed all chunks at once - faster than one by one
        embeddings = model.encode(chunks).tolist()

        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ): 
            chunk_id = f"{doc['filename']}_chunk_{chunk_counter}"

            all_chunks.append(chunk)
            all_embeddings.append(embedding)
            all_ids.append(chunk_id)
            all_metadata.append({
                "source": doc["filename"],
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

            chunk_counter += 1

            print(f"Storing {len(all_chunks)} chunks in ChromaDB...")

            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                collection.add(
                    documents= all_chunks[i:i+batch_size],
                    embeddings=all_embeddings[i:i+batch_size],
                    ids=all_ids[i:i+batch_size],
                    metadatas=all_metadata[i:i+batch_size]
                )
                print(f"Batch{i//batch_size + 1} stored")
            print("Ingestion Complete")    

if __name__ == "__main__ ":
    ingest_documents()





