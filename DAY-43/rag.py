from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

loader = TextLoader("sample_document.txt")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60
)

chunks = splitter.split_documents(documents)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

print("\nDocument Q&A System Ready\n")

while True:

    question = input("Ask a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    response = qa_chain.invoke(question)

    print("\nAnswer:")
    print(response["result"])
    print("\n" + "-" * 60 + "\n")