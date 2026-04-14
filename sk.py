from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "I'm shrishti Yadav",
    "I'm learning AI ML" 
]

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(docs)

similarity = cosine_similarity(vectors[0:1], vectors[1:2])

print("Cosine Similarity:", similarity[0][0])