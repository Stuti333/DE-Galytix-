import csv
import gensim
import pandas as pd
from word_embedding import WordEmbedding
from cosine_similarity import SimilarityCalculator
import numpy as np

data = pd.read_csv('phrases.csv', encoding='latin-1')
df = pd.DataFrame(data)
wv = gensim.models.KeyedVectors.load_word2vec_format('vectors.csv', binary=False)

word_embedding = WordEmbedding(wv)
vectors = np.array(df['Phrases'].apply(word_embedding.generate_word_embedding).to_list())

similarity_calculator = SimilarityCalculator()
cosine_similarity = similarity_calculator.calculate_cosine_similarity(vectors)
cosine_similarity_df = pd.DataFrame(cosine_similarity, columns=df['Phrases'], index=df['Phrases'])

print("Cosine Similarity:")
print(cosine_similarity_df)


