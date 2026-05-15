from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

client = OpenAI(api_key=os.getenv("************************"))

evaluation_questions = [
    {
        "question": "What is Retrieval-Augmented Generation (RAG)?",
        "expected_keywords": ["retrieval", "context", "documents", "llm"]
    },
    {
        "question": "Why is prompt engineering important in LLM applications?",
        "expected_keywords": ["instructions", "accuracy", "context", "responses"]
    },
    {
        "question": "What is conversational memory in AI systems?",
        "expected_keywords": ["history", "context", "conversation", "memory"]
    },
    {
        "question": "Why are embeddings used in semantic search?",
        "expected_keywords": ["vectors", "semantic", "similarity", "meaning"]
    }
]

results = []

print("\nGENAI WEEKLY EVALUATION\n")

for idx, item in enumerate(evaluation_questions, start=1):

    print(f"\nQuestion {idx}: {item['question']}")
    user_answer = input("Your Answer: ")

    keyword_score = sum(
        1 for keyword in item["expected_keywords"]
        if keyword.lower() in user_answer.lower()
    )

    evaluation_prompt = f"""
    Evaluate the following learner answer.

    Question:
    {item['question']}

    Answer:
    {user_answer}

    Give:
    1. Score out of 10
    2. One strength
    3. One improvement suggestion
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI engineering mentor."
            },
            {
                "role": "user",
                "content": evaluation_prompt
            }
        ],
        temperature=0.3
    )

    feedback = response.choices[0].message.content

    results.append({
        "question": item["question"],
        "answer": user_answer,
        "keyword_score": keyword_score,
        "feedback": feedback
    })

    print("\nEvaluation Feedback:")
    print(feedback)

with open("weekly_evaluation_results.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nEvaluation completed successfully.")
print("Results saved in weekly_evaluation_results.json")