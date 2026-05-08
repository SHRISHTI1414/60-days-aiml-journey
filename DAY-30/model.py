from openai import OpenAI

client = OpenAI(
    api_key="skp***********************"
)
content = """
Remote work has changed how companies operate across the world.
Organizations are increasingly adopting flexible work environments,
allowing employees to collaborate digitally instead of working from
traditional office spaces. This shift has improved accessibility,
reduced commuting time, and enabled global collaboration, but it has
also introduced challenges related to communication, productivity,
and maintaining team culture.
"""
summary_prompt = f"""
Summarize the following passage in 3 concise points:

{content}
"""

summary = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are an expert summarization assistant."
        },
        {
            "role": "user",
            "content": summary_prompt
        }
    ],
    temperature=0.4,
    max_tokens=120
)

qa_prompt = f"""
Read the following passage and answer the question clearly.

Passage:
{content}

Question:
What are the major advantages of remote work?
"""

qa = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI question-answering assistant."
        },
        {
            "role": "user",
            "content": qa_prompt
        }
    ],
    temperature=0.3,
    max_tokens=100
)

print("SUMMARY:\n")
print(summary.choices[0].message.content)

print("\nQUESTION ANSWERING:\n")
print(qa.choices[0].message.content)