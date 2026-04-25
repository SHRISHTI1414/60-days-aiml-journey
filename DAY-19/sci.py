import numpy as np
 
vec1 = np.array([3.0, 4.0, 0.0])
vec2 = np.array([1.0, 2.0, 2.0])
 
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

raw_similarity = cosine_similarity(vec1, vec2)
 
def normalize(v):
    return v / (np.linalg.norm(v) + 1e-10)

vec1_norm = normalize(vec1)
vec2_norm = normalize(vec2)
 
normalized_similarity = np.dot(vec1_norm, vec2_norm)
 
print("Raw Similarity:", raw_similarity)
print("Normalized Similarity:", normalized_similarity)

print("\nNormalized Vectors:")
print("vec1_norm:", vec1_norm)
print("vec2_norm:", vec2_norm)