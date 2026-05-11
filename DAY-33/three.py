from langchain_openai import ChatOpenAI
from langchain import ConversationBufferMemory
from langchain  import ConversationChain
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.4
)

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

response1 = conversation.predict(
    input="Hi, I am Shrishti and I am learning AI Engineering."
)

response2 = conversation.predict(
    input="Can you suggest beginner-friendly AI projects for me?"
)

response3 = conversation.predict(
    input="Also remember that I enjoy working on NLP-based applications."
)

response4 = conversation.predict(
    input="Based on our conversation, what kind of projects would suit me?"
)

print("\nBOT RESPONSE 1:\n")
print(response1)

print("\nBOT RESPONSE 2:\n")
print(response2)

print("\nBOT RESPONSE 3:\n")
print(response3)

print("\nBOT RESPONSE 4:\n")
print(response4)