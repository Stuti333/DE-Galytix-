import numpy as np
from sklearn.metrics.pairwise import cosine_distances

class SimilarityCalculator:
    @staticmethod
    def calculate_cosine_similarity(vectors):
        if len(vectors) == 0:
            return np.array([])

        if len(vectors[0].shape) == 1:
            vectors = [vector.reshape(1, -1) for vector in vectors]

        concatenated_vectors = np.concatenate(vectors, axis=0)
        cosine_similarity = 1 - cosine_distances(concatenated_vectors)
        return cosine_similarity