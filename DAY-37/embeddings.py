from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

documents = [
    "Large Language Models can generate human-like text responses.",
    "Vector embeddings help AI systems understand semantic meaning.",
    "FastAPI is commonly used for deploying AI applications.",
    "Semantic search retrieves information based on contextual similarity.",
    "Conversation memory improves chatbot interactions."
]

def generate_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

document_embeddings = []

print("\nGenerating embeddings...\n")

for doc in documents:
    embedding = generate_embedding(doc)

    document_embeddings.append({
        "text": doc,
        "embedding": embedding
    })

    print(f"Processed: {doc[:60]}")

query = input("\nEnter your semantic search query: ")

query_embedding = generate_embedding(query)

similarity_scores = []

for item in document_embeddings:

    score = cosine_similarity(
        [query_embedding],
        [item["embedding"]]
    )[0][0]

    similarity_scores.append({
        "document": item["text"],
        "similarity": float(score)
    })

ranked_results = sorted(
    similarity_scores,
    key=lambda x: x["similarity"],
    reverse=True
)

print("\nTop Semantic Matches:\n")

for result in ranked_results:
    print(f"Document: {result['document']}")
    print(f"Similarity Score: {round(result['similarity'], 4)}\n")