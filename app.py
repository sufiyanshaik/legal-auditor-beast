import os
import tempfile
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from ingest import build_advanced_retriever
from rerank import setup_reranker

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv())

# Set layout to wide for a better UI experience
st.set_page_config(page_title="The Legal Auditor", page_icon="⚖️", layout="wide")

# --- 1. INITIALIZE THE RAG PIPELINE (Now accepts dynamic files!) ---
@st.cache_resource(show_spinner="Ingesting document & spinning up the Beast...")
def initialize_pipeline(file_path):
    # We now pass the dynamic file path instead of the hardcoded SAFE note
    base_retriever = build_advanced_retriever(file_path)
    reranker_pipeline = setup_reranker(base_retriever)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return reranker_pipeline, llm

# --- 2. CITATION MODE PROMPT ---
template = """
You are a highly precise Legal Auditor. Your job is to answer the user's question based strictly on the provided contract excerpts.
If the answer is not contained in the context, you must reply: "I cannot find this information in the provided contract." Do not hallucinate.

When you provide an answer, you MUST include a "CITATION" section at the end, quoting the exact snippet of text you used to formulate your answer.

Context Provided from Contract:
{context}

User Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(f"Snippet: {doc.page_content}" for doc in docs)

# --- 3. STREAMLIT UI ---
st.title("⚖️ The Legal Auditor (Beast Mode)")
st.markdown("Upload any contract to instantly audit it using Hybrid Search & Gemini 2.5 Flash.")

# Build a clean sidebar for the file uploader
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Drop a .docx file here", type=["docx"])
    
    if uploaded_file:
        st.success(f"Successfully loaded: {uploaded_file.name}")
        st.markdown("---")
        st.markdown("**Active Architecture:**")
        st.markdown("- Embeddings: `all-MiniLM-L6-v2`")
        st.markdown("- Vector DB: `FAISS`")
        st.markdown("- Re-Ranker: `Cohere V3`")
        st.markdown("- LLM: `Gemini 2.5 Flash`")

# Main execution block
if uploaded_file is not None:
    # 🚨 THE FIX: Write the uploaded memory stream to a temporary physical file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Feed the physical temp file to your architecture
    reranker, llm = initialize_pipeline(tmp_file_path)

    # Build the execution chain
    rag_chain = (
        {"context": reranker | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Initialize chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if user_query := st.chat_input("Ask a question about the uploaded contract..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Auditing contract clauses..."):
                response = rag_chain.invoke(user_query)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    # What the user sees before they upload a file
    st.info("👈 Please upload a contract in the sidebar to begin auditing.")