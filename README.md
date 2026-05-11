## Agentic RAG
A production-grade Retreieval-Augmented Generation system with autonomous query rewriting, hybrid retrieval, reranking, self evaluation and multi-step reasoning built entirely on python.

## What Makes this Agentic
Standard RAG is passive - It takes a question, searches once, and generates an answer. This one takes in a query, analyses the query, rewrite the query, searches the query, reranks the query, evaluates the answer and generates an answer.

## Step by Step Procedure

```
User Question
      ↓
Step 1: QUERY ANALYSER
  Classifies the question:
  - Simple   → direct factual question
  - Complex  → requires multiple pieces of information
  - Multi-hop → answer depends on chaining multiple facts
      ↓
Step 2: QUERY REWRITER
  Transforms the raw question:
  - Fixes vague or ambiguous phrasing
  - Expands short queries with relevant context
  - Decomposes multi-hop questions into sub-questions
      ↓
Step 3: RETRIEVER (Hybrid)
  Chooses retrieval strategy based on query type:
  - Vector search  → semantic similarity (meaning-based)
  - Keyword search → exact term matching (precision-based)
  - Hybrid         → both combined with weighted scoring
      ↓
Step 4: RERANKER
  Scores every retrieved chunk for relevance:
  - Removes chunks below relevance threshold
  - Reorders remaining chunks — best first
  - Returns top K most relevant
      ↓
Step 5: EVALUATOR
  Asks: "Do I have enough context to answer well?"
  - If YES → proceed to generation
  - If NO  → rewrite query and retrieve again (max 3 attempts)
      ↓
Step 6: GENERATOR
  Produces the final answer:
  - Simple questions  → direct answer
  - Complex questions → multi-step reasoning chain
  - Always cites which chunks were used
```

## The Four Capabilities

### 1. Query Rewriting
```
User asks:  "how bad is turkana?"
Agent sees: Vague, too short, no context
Rewrites:   "Turkana County financial inclusion exclusion rate 
             barriers statistics Kenya 2024"
Result:     Much better retrieval 
```

### 2. Hybrid Retrieval
```
"What is the average inclusion rate?"
→ Vector search (semantic meaning matches stats chunks)

"Find the exact Turkana 2024 figure"  
→ Keyword search (exact term matching)

"Compare Nairobi and Turkana inclusion"
→ Hybrid (combines both approaches)
```

### 3. Reranking
```
Retrieved 8 chunks → Agent scores each 0.0 to 1.0
Chunk 1: 0.92 ← highly relevant
Chunk 2: 0.87 ← relevant
Chunk 3: 0.43 ← marginally relevant
Chunk 4: 0.12 ← irrelevant → removed
Returns top 3 only 
```

### 4. Self-Evaluation Loop
```
First retrieval → Evaluator: "Not enough context, retry"
Rewrites query  → Second retrieval → Evaluator: "Sufficient "
Proceeds to generation
Max 3 attempts before proceeding with best available context
```

## A Reasoning Example
In the UI, the system shows its own thinking steps taken to arrive at the final answer
```
Query Analysis
   Type: Complex
   Confidence: 0.87
   Reasoning: Question requires multiple data points

Query Rewriting  
   Original:  "which counties are struggling?"
   Rewritten: "Kenya counties lowest financial inclusion 
               rate exclusion barriers 2024"

Retrieval (Hybrid)
   Strategy: Hybrid search selected
   Chunks found: 8
   
Reranking
   Chunks after reranking: 3
   Top relevance score: 0.94

Evaluation
   Attempt: 1 of 3
   Context sufficient:  Yes

 Generation
   Mode: Multi-step reasoning
   Steps: 3

 Answer (with 3 source citations)

```
## Project Structure
```
Agentic_RAG/
│
├── core/
│   ├── query_analyser.py    # Classifies query complexity
│   ├── query_rewriter.py    # Rewrites and decomposes queries
│   ├── retriever.py         # Hybrid vector + keyword search
│   ├── reranker.py          # Scores and filters chunks
│   ├── evaluator.py         # Decides if context is sufficient
│   └── generator.py         # Reasons and generates answer
│
├── agent.py                 # Orchestrator — runs the 6-step loop
├── ingest.py                # Loads documents into ChromaDB
├── app.py                   # Streamlit UI with reasoning trace
├── requirements.txt
└── README.md
```
## Setup and Installation
### Step 1 — Create Virtual Environment
```bash
mkdir agentic-rag
cd agentic-rag
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Pull AI Model
```bash
ollama pull tinyllama
```

### Step 4 — Ingest Documents
```bash
python core/ingest.py
```

### Step 5 — Run the App
```bash
streamlit run app.py
```

## Tech Stack
1. Python => Pure Implementation, no frameworks.

2. ChromaDB => Vector Storage for fast semantic search.

3. Ollama + TinyLlama => Local LLM that is free, private and offline.

4. Sentence Transformers => Better Embeddings than TinyLlama

5. Streamlit => User Interface that shows reasoning trace visually.

6. Numpy => Reranking math for cosine similarity scoring.

## Key Concepts Explained
### Why Hybrid Search

Vector search finds semantically similar content, which is greatfor meaning-based queries

Keyword search finds exact term matches, which is great for specific names, numbers and dates.

Neither is perfect alone, Hybrid search combines both with weightes scoring(70/30) to get the best of both approaches.

### Why Reranking

The retrieval casts a wide net, it potentially finds relevant chunks, but not all retrieved chunks are equally useful. Reranking applies a second filter for noise and surface the most relevant context. It is the difference between finding a book in the library and choosing which pages to actually read.

### Why Self-Evaluation

Somethimes the first retrieval misses. The query might be ambigous. The relevanet information might use different terminology.

Self evaluation lets the agent recognise when it has insufficient context and try again with a better query.

## Why Reasoning Trace

In Production AI systems especially for high-stakes domains like financial inclusion or empployment policy, users need to see how the system reached its answer. A visible reasoning trace turns a black box into an auditable, explainable system.

## What Documents Can It work with

The system works with any text documents. For demonstration we use Kenya financial inclusion data, making it directly relevant to the previous projects.

Load any of these:
- PDF reports
- CSV files converted to text
- Plain Text documents
- Web Scraped content


## Author
Built as part of a self-directed AI engineering learning journey

Paul Gikonyo Data Analyst@Everything Data Africa

## License
MIT License