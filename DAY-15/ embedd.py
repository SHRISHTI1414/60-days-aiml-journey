import numpy as np

 
embeddings = np.array([
    [0.2, 0.1, 0.4, 0.7],   
    [0.3, 0.8, 0.2, 0.5],   
    [0.9, 0.4, 0.6, 0.1]     
])

 
sentence_embedding = np.mean(embeddings, axis=0)

print("Sentence Embedding:", sentence_embedding)

 
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

 
s1 = np.array([0.2, 0.3, 0.5, 0.7])
s2 = np.array([0.1, 0.4, 0.6, 0.8])

print("Similarity:", cosine_similarity(s1, s2))