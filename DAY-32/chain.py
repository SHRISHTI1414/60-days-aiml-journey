from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("************"),
    temperature=0.3
)


research_document = """
Large Language Models (LLMs) are transforming the way modern software systems interact with information.
Instead of relying only on predefined rules, LLMs can reason over natural language, summarize documents,
answer questions, and generate human-like responses.

However, real-world enterprise systems face several challenges while integrating LLMs into production:
high latency, hallucinations, context-window limitations, and unreliable outputs.

To address these issues, modern AI engineering workflows combine retrieval systems, vector databases,
prompt engineering, and orchestration frameworks like LangChain. These systems break large documents into chunks,
retrieve relevant context dynamically, and generate grounded responses using external knowledge.

This architecture is commonly used in Retrieval-Augmented Generation (RAG) systems, AI copilots,
research assistants, legal document analyzers, and enterprise search applications.

As AI systems scale, workflow orchestration, modular pipelines, observability, and evaluation become critical
for building reliable and production-ready GenAI applications.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(research_document)

prompt = ChatPromptTemplate.from_template(
    """
    You are an AI research assistant.

    Analyze the following document chunk and generate:
    1. A concise summary
    2. Key technical concepts
    3. Real-world applications

    Document:
    {chunk}
    """
)

parser = StrOutputParser()

chain = prompt | llm | parser

for index, chunk in enumerate(chunks, start=1):
    response = chain.invoke({"chunk": chunk})

    print(f"\n{'='*60}")
    print(f"DOCUMENT CHUNK {index}")
    print(f"{'='*60}\n")

    print(response)