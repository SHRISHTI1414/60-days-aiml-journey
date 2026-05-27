spam_keywords = [
    "free",
    "win",
    "offer",
    "money",
    "urgent",
    "click",
    "prize",
    "lottery",
    "claim",
    "cash"
]

def classify_message(message):

    message = message.lower()

    spam_score = 0

    matched_keywords = []

    for keyword in spam_keywords:

        if keyword in message:

            spam_score += 1
            matched_keywords.append(keyword)

    if spam_score >= 2:
        result = "SPAM"

    else:
        result = "NOT SPAM"

    return {
        "message": message,
        "classification": result,
        "spam_score": spam_score,
        "matched_keywords": matched_keywords
    }

while True:

    user_input = input("\nEnter Message (or type exit): ")

    if user_input.lower() == "exit":
        break

    output = classify_message(user_input)

    print("\nClassification Result")
    print(output)