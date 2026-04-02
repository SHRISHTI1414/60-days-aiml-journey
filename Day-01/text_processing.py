import string
from collections import Counter

# Basic stopwords list (you can expand this)
STOPWORDS = {
    "the", "is", "in", "and", "to", "of", "a", "for", "on", "with",
    "as", "by", "an", "be", "this", "that", "it"
}

def tokenize(text):
    """
    Convert text into lowercase words and remove punctuation
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return tokens

def remove_stopwords(tokens):
    """
    Remove common stopwords from token list
    """
    return [word for word in tokens if word not in STOPWORDS]

def word_frequency(tokens):
    """
    Count frequency of each word
    """
    return Counter(tokens)


# Example usage
if __name__ == "__main__":
    paragraph = """
    AI is transforming the world and the way we solve problems.
    Learning Python for AI is important and useful.
    """

    tokens = tokenize(paragraph)
    filtered_tokens = remove_stopwords(tokens)
    freq = word_frequency(filtered_tokens)

    print("Tokens:", tokens)
    print("\nAfter Stopword Removal:", filtered_tokens)
    print("\nWord Frequency:", freq)