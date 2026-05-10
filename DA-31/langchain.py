from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key="******************"
)

prompt = ChatPromptTemplate.from_template(
    "Explain the importance of {topic} in AI engineering in 4 concise points."
)

parser = StrOutputParser()

chain = prompt | llm | parser

response = chain.invoke(
    {"topic": "LangChain workflows"}
)

print(response)