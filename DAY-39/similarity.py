from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

documents = [
    {
        "title": "RAG Systems",
        "content": "Retrieval-Augmented Generation combines retrieval pipelines with language models."
    },
    {
        "title": "Embeddings",
        "content": "Embeddings convert text into dense vector representations capturing semantic meaning."
    },
    {
        "title": "Vector Databases",
        "content": "FAISS enables efficient similarity search across high-dimensional vectors."
    },
    {
        "title": "Prompt Engineering",
        "content": "Prompt structure directly impacts LLM response quality and reasoning."
    },
    {
        "title": "AI Agents",
        "content": "AI agents use tools, memory, and reasoning to perform dynamic workflows."
    }
]

def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

print("\nGenerating document embeddings...\n")

embedding_vectors = []

for doc in documents:

    combined_text = f"{doc['title']} {doc['content']}"

    embedding = generate_embedding(combined_text)

    embedding_vectors.append(embedding)

    print(f"Indexed: {doc['title']}")

embedding_array = np.array(embedding_vectors).astype("float32")

dimension = embedding_array.shape[1]

index = faiss.IndexFlatIP(dimension)

faiss.normalize_L2(embedding_array)

index.add(embedding_array)

print(f"\nDocuments indexed successfully: {index.ntotal}")

query = input("\nEnter your semantic search query: ")

query_embedding = generate_embedding(query)

query_array = np.array([query_embedding]).astype("float32")

faiss.normalize_L2(query_array)

top_k = 3

scores, indices = index.search(query_array, top_k)

print("\nTop Semantic Search Results:\n")

for rank, idx in enumerate(indices[0]):

    result = documents[idx]

    similarity_score = scores[0][rank]

    print(f"Rank: {rank + 1}")
    print(f"Title: {result['title']}")
    print(f"Content: {result['content']}")
    print(f"Similarity Score: {round(float(similarity_score), 4)}\n")