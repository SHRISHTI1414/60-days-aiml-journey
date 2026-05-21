from langchain_openai import OpenAIEmbeddings
from langchain import FAISS
from langchain  import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

documents = [
    Document(
        page_content="""
        Retrieval-Augmented Generation improves LLM responses
        by retrieving relevant external context before generation.
        """
    ),
    Document(
        page_content="""
        FAISS is a vector similarity search library optimized
        for high-dimensional embedding retrieval workflows.
        """
    ),
    Document(
        page_content="""
        Embeddings convert semantic meaning into numerical vector space representations.
        """
    ),
    Document(
        page_content="""
        Chunk overlap helps preserve contextual continuity
        between document segments during retrieval.
        """
    ),
    Document(
        page_content="""
        Smaller chunk sizes improve retrieval precision
        while larger chunks improve contextual understanding.
        """
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=30
)

split_docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = FAISS.from_documents(
    split_docs,
    embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)

print("\nOptimized RAG Retrieval System Ready\n")

query = input("Enter your query: ")

retrieved_docs = retriever.invoke(query)

query_embedding = embeddings.embed_query(query)

print("\nTop Retrieved Chunks:\n")

for idx, doc in enumerate(retrieved_docs, start=1):

    doc_embedding = embeddings.embed_query(doc.page_content)

    similarity_score = cosine_similarity(
        [query_embedding],
        [doc_embedding]
    )[0][0]

    print(f"Result {idx}")
    print(f"Similarity Score: {round(float(similarity_score), 4)}")
    print(f"Content: {doc.page_content.strip()}\n")

print("-" * 60)

print("\nOptimization Techniques Applied:\n")

print("1. Reduced chunk size for better retrieval precision")
print("2. Added chunk overlap for contextual continuity")
print("3. Used semantic similarity search")
print("4. Tuned top-k retrieval settings")
print("5. Compared similarity scores for evaluation")