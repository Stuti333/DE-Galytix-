import gensim
import numpy as np

class WordEmbedding:
    def __init__(self, wv):
        self.wv = wv

    def generate_word_embedding(self, text):
        tokens = gensim.utils.simple_preprocess(text)
        if not tokens:
            return np.zeros(self.wv.vector_size)
        
        word_embeddings = [self.wv[word] for word in tokens if word in self.wv]
        
        if not word_embeddings:
            return np.zeros(self.wv.vector_size)
        
        vectors = np.sum(word_embeddings, axis=0)
        vectors /= np.linalg.norm(vectors)
        
        return vectors