import numpy as np
from scipy.spatial.distance import cosine

class EmbeddingSimilarity:
    def __init__(self):
        pass

    
    def dot_product(self, vec1, vec2):
        return np.dot(vec1, vec2)
 
    def cosine_similarity(self, vec1, vec2):
        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm_v1 * norm_v2)

    
    def cosine_scipy(self, vec1, vec2):
        return 1 - cosine(vec1, vec2)

    
    def normalize(self, vec):
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    
    def compare(self, vec1, vec2):
        results = {
            "Dot Product": self.dot_product(vec1, vec2),
            "Cosine (Manual)": self.cosine_similarity(vec1, vec2),
            "Cosine (SciPy)": self.cosine_scipy(vec1, vec2)
        }
        return results


 
if __name__ == "__main__":
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([2.0, 4.0, 6.0])

    sim = EmbeddingSimilarity()
    result = sim.compare(vec1, vec2)

    print("Similarity Results:")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")