from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

documents = [
    "Large Language Models generate human-like responses.",
    "FAISS enables fast vector similarity search.",
    "Embeddings capture semantic meaning of text.",
    "RAG systems combine retrieval and generation.",
    "Vector databases are important in modern AI systems."
]

def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

print("\nGenerating embeddings...\n")

embedding_list = []

for doc in documents:
    embedding = generate_embedding(doc)

    embedding_list.append(embedding)

    print(f"Processed: {doc}")

embedding_array = np.array(embedding_list).astype("float32")

dimension = embedding_array.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embedding_array)

print(f"\nTotal vectors stored in FAISS: {index.ntotal}")

query = input("\nEnter your search query: ")

query_embedding = generate_embedding(query)

query_array = np.array([query_embedding]).astype("float32")

k = 3

distances, indices = index.search(query_array, k)

print("\nTop Similar Documents:\n")

for i, idx in enumerate(indices[0]):

    print(f"Rank {i+1}")
    print(f"Document: {documents[idx]}")
    print(f"Distance Score: {distances[0][i]}\n")