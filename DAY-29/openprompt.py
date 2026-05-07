from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

prompt = "Explain how Large Language Models work in simple words."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=150
)

print("Prompt:\n", prompt)
print("\nGenerated Response:\n")
print(response.choices[0].message.content)