from dotenv import load_dotenv
from openai import OpenAI
from langchain import ConversationBufferMemory
from datetime import datetime
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("***********************"))

memory = ConversationBufferMemory(return_messages=True)

SYSTEM_PROMPT = """
You are an intelligent AI mentor chatbot.
You remember previous conversations and respond contextually.
Keep responses concise, practical, and conversational.
"""

def chat(user_input):
    history = memory.load_memory_variables({})["history"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history:
        role = "user" if msg.type == "human" else "assistant"
        messages.append({
            "role": role,
            "content": msg.content
        })

    messages.append({
        "role": "user",
        "content": user_input
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )

    reply = response.choices[0].message.content

    memory.save_context(
        {"input": user_input},
        {"output": reply}
    )

    return reply

print("\nAI Memory Chatbot Started")
print("Type 'exit' to stop\n")

while True:
    user_query = input("You: ")

    if user_query.lower() == "exit":
        print("\nSession Ended")
        break

    bot_reply = chat(user_query)

    print(f"\nBot: {bot_reply}\n")

    with open("chat_history.txt", "a") as file:
        file.write(
            f"\n[{datetime.now()}]\nUser: {user_query}\nBot: {bot_reply}\n"
        )