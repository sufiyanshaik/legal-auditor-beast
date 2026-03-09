# ⚖️ The Legal Auditor (Enterprise RAG Pipeline)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://legal-auditor-beast-q98t6pzt96fkondvgt6og3.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![LangChain](https://img.shields.io/badge/LangChain-⚡-green.svg)](https://python.langchain.com/)

A production-ready, hallucination-free Retrieval-Augmented Generation (RAG) web application engineered to audit complex legal contracts and financial documents. 

**Live Demo:** [Launch The Legal Auditor](https://legal-auditor-beast-q98t6pzt96fkondvgt6og3.streamlit.app/)

---

## 💡 The Business Problem
Large Language Models (LLMs) are incredible synthesizers, but in high-stakes domains (legal, financial reconciliation, compliance), a "hallucinated" answer is a massive liability. Auditors and attorneys do not trust black-box AI; they require exact, verifiable textual receipts.

**The Solution:** This architecture moves past basic LLM wrappers by implementing a custom Hybrid Search pipeline, Cross-Encoder Re-Ranking, and strict "Citation Mode" prompt engineering to guarantee answers are mathematically backed by the source document.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[📄 User Uploads Contract] --> B[⚙️ Docx Loader & Text Splitter]
    B -->|Parent & Child Chunks| C[🧠 Local Embeddings: HuggingFace all-MiniLM]
    C --> D[(🗄️ FAISS Vector Database)]
    
    E[👤 User Asks Question] --> C
    
    D -->|Retrieves Top 10 Docs| F[🎯 Cohere V3 Re-Ranker]
    F -->|Filters Top 3 Docs| G[🤖 Google Gemini 2.5 Flash]
    
    G -->|Applies Citation Prompt| H[💻 Streamlit UI: Answer + Receipt]
    
    style A fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px
    style D fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    style G fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    style H fill:#fce4ec,stroke:#e91e63,stroke-width:2px