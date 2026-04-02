import string
from collections import Counter

# basic stopwords
stopwords = {
    "the", "is", "in", "and", "to", "of", "a", "for", "on", "with",
    "as", "by", "an", "be", "this", "that", "it"
}

# function to split text into words
def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return words

# remove common words
def remove_stopwords(words):
    filtered = []
    for word in words:
        if word not in stopwords:
            filtered.append(word)
    return filtered

# count frequency
def word_frequency(words):
    freq = Counter(words)
    return freq


# testing
text = "AI is transforming the world and the way we solve problems. Learning python is useful."

tokens = tokenize(text)
clean_words = remove_stopwords(tokens)
freq = word_frequency(clean_words)

print("Tokens:", tokens)
print("\nAfter removing stopwords:", clean_words)
print("\nWord frequency:", freq)