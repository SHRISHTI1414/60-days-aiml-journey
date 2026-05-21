from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain  import RecursiveCharacterTextSplitter
from langchain  import FAISS
from langchain  import RetrievalQA
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

documents = [
    Document(
        page_content="""
        Retrieval-Augmented Generation combines retrieval systems with language models
        to generate grounded and context-aware responses.
        """
    ),
    Document(
        page_content="""
        Embeddings convert text into numerical vector representations that capture semantic meaning.
        """
    ),
    Document(
        page_content="""
        Vector databases like FAISS enable efficient similarity search across embeddings.
        """
    ),
    Document(
        page_content="""
        LangChain helps developers build orchestration pipelines for LLM applications.
        """
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
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
    search_kwargs={"k": 2}
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

print("\nRAG Pipeline Ready\n")

while True:

    query = input("Ask a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    response = rag_chain.invoke(query)

    print("\nAnswer:")
    print(response["result"])
    print("\n" + "-" * 50 + "\n")