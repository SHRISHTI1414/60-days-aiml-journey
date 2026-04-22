import numpy as np
from collections import Counter
from scipy.stats import entropy

 
text = "AI is transforming the world AI is powerful and AI is the future"

 
words = text.lower().split()

 
word_counts = Counter(words)
total_words = sum(word_counts.values())

 
word_prob = {word: count / total_words for word, count in word_counts.items()}

print("Word Probabilities:")
for word, prob in word_prob.items():
    print(f"{word}: {prob:.3f}")

 
prob_values = list(word_prob.values())
entropy_value = entropy(prob_values)

print("\nEntropy of distribution:", entropy_value)