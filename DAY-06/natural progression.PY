import re
import string
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

 
text = """AI is transforming industries. AI is creating new opportunities.
Machine learning and AI are shaping the future of technology."""

 
text = text.lower()
text = re.sub(r'\d+', '', text)
text = text.translate(str.maketrans('', '', string.punctuation))

 
words = text.split()
 
freq = Counter(words)
 
top_words = freq.most_common(20)

words_list = [w[0] for w in top_words]
counts = [w[1] for w in top_words]

 
counts_np = np.array(counts)

 
plt.figure(figsize=(10,5))
plt.bar(words_list, counts_np)
plt.xticks(rotation=45)
plt.title("Top Word Frequencies")
plt.xlabel("Words")
plt.ylabel("Count")
plt.tight_layout()

plt.show()