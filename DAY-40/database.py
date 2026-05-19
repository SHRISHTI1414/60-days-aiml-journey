from openai import OpenAI
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("*******************"))

mongo_client = MongoClient(os.getenv("****************************************"))

db = mongo_client["ai_engineering"]

collection = db["documents"]

documents = [
    {
        "title": "RAG Systems",
        "category": "GenAI",
        "content": "Retrieval-Augmented Generation combines retrieval systems with LLMs."
    },
    {
        "title": "Embeddings",
        "category": "AI Foundations",
        "content": "Embeddings convert text into dense vector representations."
    },
    {
        "title": "Vector Databases",
        "category": "AI Systems",
        "content": "FAISS enables efficient semantic similarity search."
    }
]

def generate_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

print("\nGenerating embeddings and storing documents...\n")

for doc in documents:

    combined_text = f"{doc['title']} {doc['content']}"

    embedding = generate_embedding(combined_text)

    document_data = {
        "title": doc["title"],
        "category": doc["category"],
        "content": doc["content"],
        "embedding": embedding,
        "created_at": datetime.utcnow()
    }

    collection.insert_one(document_data)

    print(f"Stored: {doc['title']}")

print("\nDocuments stored successfully in MongoDB")

print("\nStored Documents:\n")

for item in collection.find({}, {"embedding": 0}):

    print(f"Title: {item['title']}")
    print(f"Category: {item['category']}")
    print(f"Content: {item['content']}\n")