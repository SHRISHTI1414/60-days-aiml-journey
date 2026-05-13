from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from dotenv import load_dotenv
import os
import datetime

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

def calculate_square(number: str) -> str:
    value = float(number)
    return f"The square of {value} is {value ** 2}"

def get_current_time(_: str) -> str:
    current_time = datetime.datetime.now()
    return f"Current system time: {current_time.strftime('%H:%M:%S')}"

tools = [
    Tool(
        name="Square Calculator",
        func=calculate_square,
        description="Use this tool to calculate the square of a number."
    ),
    Tool(
        name="Current Time",
        func=get_current_time,
        description="Use this tool to get the current system time."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response_1 = agent.run(
    "What is the square of 18?"
)

response_2 = agent.run(
    "Can you also tell me the current time?"
)

print("\nRESPONSE 1:\n")
print(response_1)

print("\nRESPONSE 2:\n")
print(response_2)