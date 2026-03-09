import os
from dotenv import load_dotenv, find_dotenv

from ingest import build_advanced_retriever


from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever


load_dotenv(find_dotenv())

def setup_reranker(base_retriever):

    compressor = CohereRerank(
        cohere_api_key = os.environ.get("COHERE_API_KEY"),
        model="rerank-english-v3.0",
        top_n=3
    )

    compresson_retriever = ContextualCompressionRetriever(
        base_compressor = compressor,
        base_retriever=base_retriever
    )

    return compresson_retriever



if __name__ == "__main__":

    file_path="data/safe_note.docx"

    print("--- 1. BUILDING BASE RETRIEVER ---")
    base_retriever = build_advanced_retriever(file_path)

    print("\n--- 2. ADDING THE COHERE RE-RANKER ---")
    reranker_pipeline = setup_reranker(base_retriever)


    query = "What happens to my shares during a Liquidity Event?"
    print(f"\nSearching for: '{query}'")

    reranked_docs= reranker_pipeline.invoke(query)

    print("\n--- THE FINAL RE-RANKED CHUNKS ---")
    for i, doc in enumerate(reranked_docs):

        score = doc.metadata.get('relevance_score', 'N/A')
        print(f"\n🏆 Rank {i+1} | Relevance Score: {score}")
        print(f"Text snippet: {doc.page_content[:200]}...\n")

