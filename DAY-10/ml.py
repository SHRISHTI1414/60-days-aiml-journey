from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "AI is transforming the world",
    "Machine learning is powerful",
    "AI makes machines intelligent"
]

# Bag of Words
bow = CountVectorizer()
bow_matrix = bow.fit_transform(corpus)

print("Bag of Words:")
print(bow_matrix.toarray())
print("Features:", bow.get_feature_names_out())

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus)

print("\nTF-IDF:")
print(tfidf_matrix.toarray())
print("Features:", tfidf.get_feature_names_out())