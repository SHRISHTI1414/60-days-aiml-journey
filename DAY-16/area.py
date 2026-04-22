import numpy as np

embedding_a = np.array([0.2, 0.5, 0.1, 0.7])
embedding_b = np.array([0.3, 0.1, 0.4, 0.9])
 
dot_product = np.dot(embedding_a, embedding_b)

 
embedding_matrix = np.array([
    [0.2, 0.5, 0.1, 0.7],
    [0.3, 0.1, 0.4, 0.9],
    [0.6, 0.8, 0.2, 0.3]
])

 
transformation_matrix = np.array([
    [0.5, 0.2],
    [0.1, 0.7],
    [0.3, 0.6],
    [0.9, 0.4]
])

matrix_result = np.matmul(embedding_matrix, transformation_matrix)

print("Dot Product (Similarity):", dot_product)
print("\nMatrix Multiplication Result:\n", matrix_result)