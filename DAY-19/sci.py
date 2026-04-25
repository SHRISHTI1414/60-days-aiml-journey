import numpy as np
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
doc1 = "AI is transforming the world with intelligent systems"
doc2 = "Artificial intelligence is changing the world"
 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([doc1, doc2]).toarray()

A = X[0]  
B = X[1]   

 
def cost(x):
    similarity = np.dot(A, x) / (np.linalg.norm(A) * np.linalg.norm(x) + 1e-10)
    return 1 - similarity  # minimize

 
x0 = B.copy()  

result = minimize(cost, x0, method='L-BFGS-B')

optimized_x = result.x
final_similarity = 1 - result.fun
 
print("Initial Similarity:", cosine_similarity([A], [B])[0][0])
print("Optimized Similarity:", final_similarity)