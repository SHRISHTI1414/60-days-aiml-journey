import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

text = """AI is transforming the world. It is changing how we solve problems.
Learning NLP is important for building intelligent systems."""

sentences = sent_tokenize(text)
words = word_tokenize(text)

print("\nSentences:\n", sentences)
print("\nWords:\n", words)
print("\nTotal sentences:", len(sentences))
print("Total words:", len(words))