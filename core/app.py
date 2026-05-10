"""
This is the Final UI
"""
import streamlit as st
import sys
import os
from datetime import datetime

#Add project root to path
from agent import AgenticRAG, AgentTrace

st.set_page_config(
    page_title="Agentic_RAG",
    layout="wide"
)
# Initialize session state keys
if "query" not in st.session_state:
    st.session_state["query"] = ""

st.title("Agentic RAG -  Built From Python")
st.markdown("""
An intelligent RAG system that thinks before it answers.
Ask any question and watch the agent analyse, rewrite, retrieve, rerank, evaluate and reason step by step
""")

@st.cache_resource
def load_agent():
    """
    Laod the agent once and cache it.
    @st.cache_resource keeps heavy objects in memory between reruns, perfect for ML models and DB connections, agent has ML models inside

    """
    return AgenticRAG()
agent = load_agent()

st.sidebar.title("Agent Settings")

st.sidebar.markdown("Knowledge Base")
st.sidebar.markdown("Documents in 'documents' folder")

if st.sidebar.button("Re-ingest Documents"):
    with st.spinner("Re-ingesting documents..."):
        try:
            from ingest import ingest_documents
            ingest_documents()
            st.sidebar.success("Documents re-ingested")
            st.cache_resource.clear()
        except Exception as e:
            st.sidebar.error(f"{e}")

st.sidebar.markdown("---")
st.sidebar.markdown("Agent Components")
st.sidebar.markdown(" Query Analyser")
st.sidebar.markdown(" Query Rewriter")
st.sidebar.markdown(" Hybrid Retriever")
st.sidebar.markdown(" Reranker")
st.sidebar.markdown(" Context Evaluator")
st.sidebar.markdown(" Generator")
st.sidebar.markdown("---")
st.sidebar.markdown("Example Questions")
st.sidebar.markdown("- What is Kenya's inclusion rate?")
st.sidebar.markdown("- Why is Turkana excluded?")
st.sidebar.markdown("- Compare gender gap 2006 vs 2024")
st.sidebar.markdown("- What products suit rural Kenyans?")   

# Query Input
st.subheader("Ask the Agent")

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "Your question:",
        placeholder="e.g Why is Turkana's Financial inclusion rate so low?",
        label_visibility="collapsed",
        value=st.session_state.query
    )

with col2:
    ask_button = st.button("Ask Agent", type = "primary")



# Example question buttons
st.markdown("Quick Examples")

ex_col1, ex_col2, ex_col3 = st.columns(3)

with ex_col1:
    if st.button("Inclusion rate"):
        query = "What is Kenya's financial inclusion rate?"
        ask_button = True

with ex_col2:
    if st.button("Gender Gap"):
        query = "Compare the gender gap in 2006 and 2024" 
        ask_button = True

with ex_col3:
    if st.button("County Analysis"):
        query = "Why is Turkana's inclusion rate lower than Nairobi?"
        ask_button = True  

if ask_button and query:
    st.session_state[query] = ""

    # Run the agent
    with st.spinner("Agent Thinking..."):
        trace = agent.ask(query)

    # Reasoning Trace
    st.subheader("Agent Reasoning Trace")

    #Step Indicators
    steps = st.columns(6)
    step_labels = [
        (" - ", "Analysing"),
        (" - ", "Rewriting"),
        (" - ", "Retrieving"),
        (" - ", "Reranking"),
        (" - ", "Evaluating"),
        (" - ", "Generating")
    ]  
           

    st.markdown("")

    #Detailed trace in expanders
    with st.expander("Step 1 - Query Analysis", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Query Type", trace.query_type.upper())
        with col2:
            st.metric("Complexity", f"{trace.complexity_score:.0%}")
        with col3:
            st.metric("Sub-questions", len(trace.sub_questions))   

        if trace.sub_questions:
            st.markdown("Decomposed into:")
            for i, sq in enumerate(trace.sub_questions):
                st.markdown(f" {i+1}.{sq}") 

    with st.expander("Step 2 - Query Rewriting"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("Original:")
            st.info(trace.original_query)   

        with col2:
            st.markdown("Rewritten")
            if trace.rewrite_applied:
                st.success(trace.rewritten_query)
            else:
                st.info(trace.rewritten_query + "no change needed")  

    with st.expander("Step 3 - Retrieval"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Strategy used", trace.retrieval_strategy.upper())

        with col2:
            st.metric("Chunks Retrieved", trace.chunks_retrieved) 

    with st.expander("Step 4 - Reranking"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks before", trace.chunks_retrieved)
        with col2:
            st.metric("Chunks After", trace.chunks_after_reranking)  

        removed = trace.chunks_retrieved - trace.chunks_after_reranking

        if removed > 0:
            st.warning(f"{removed} chunks removed as irrelevant")
        else:
            st.success("All chunks passed relevance threshold")

    with st.expander("Step 5 - Context Evaluation"):
        col1, col2 = st.columns(2)
        with col1: 
            st.metric("Attempts", trace.evaluation_attempts)

        with col2:
            passed = "Yes" if trace.evaluation_passed else "Partial"
            st.metric("Context Sufficient", passed)   

        if trace.evaluation_attempts > 1:
            st.warning(
                f"Required {trace.evaluation_attempts} retrieval attempts"
                f"question may be outside knowledge base"
            )  

    with st.expander("Step 6 - Generation"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mode", trace.generation_mode.upper())
        with col2:
            st.metric("Tokens Used", trace.tokens_used)
        with col3:
            st.metric("Time", f"{trace.total_time_seconds}s")

        if trace.reasoning_steps:
            st.markdown("Reasoning Steps:")
            for step in trace.reasoning_steps:
                st.markdown(f"{step}")     

    # The Answer
    st.markdown("---")
    st.subheader("Answer")

    if trace.success:
        st.success(trace.answer)
    else:
        st.error(trace.answer)  


    # Citations
    if trace.citations:
        st.markdown("Sources: ")
        for citation in trace.citations:
            st.markdown(f" - {citation}")   

    # Performance Summary
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

    with perf_col1:
        st.metric("Total Time", f"{trace.total_time_seconds}s")

    with perf_col2:
        st.metric("Tokens Used", trace.tokens_used)

    with perf_col3:
        st.metric("Retrieval Attempts", trace.evaluation_attempts)

    with perf_col4:
        status = "Success" if trace.success else "Failed"
        st.metric("Status", status) 

elif ask_button and not query:
    st.warning("Please type a question first!")

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:12px'>
 Agentic RAG — Built From Scratch |
Query Analysis · Query Rewriting · Hybrid Retrieval ·
Reranking · Self-Evaluation · Multi-Step Reasoning
</div>



""", unsafe_allow_html=True)



