import csv
import gensim
from gensim.utils import simple_preprocess
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

data = pd.read_csv('phrases.csv', encoding='latin-1')
df = pd.DataFrame(data)
wv = KeyedVectors.load_word2vec_format('vectors.csv', binary=False)

print(df.head())

def word2vec_embedding(text):
    tokens = gensim.utils.simple_preprocess(text)
    if not tokens:
        return np.zeros(wv.vector_size)
    
    # Get the word embeddings for each token
    word_embeddings = [wv[word] for word in tokens if word in wv]
    
    if not word_embeddings:
        return np.zeros(wv.vector_size)
    
    # Calculate the normalized sum of word embeddings
    phrase_vector = np.sum(word_embeddings, axis=0)
    phrase_vector /= np.linalg.norm(phrase_vector)
    
    return phrase_vector

df['Word2vec_Embedding'] = df['Phrases'].apply(word2vec_embedding)

print(df[['Phrases', 'Word2vec_Embedding']])

vectors = np.array(df['Word2vec_Embedding'].to_list())

def cal_cosine_similarity(vectors):
    if len(vectors) == 0:
        return np.array([])

    # Ensure all vectors are NumPy arrays
    vectors = [np.asarray(vector) for vector in vectors]

    if len(vectors[0].shape) == 1:
        vectors = [vector.reshape(1, -1) for vector in vectors]

    concatenated_vectors = np.concatenate(vectors, axis=0)
    cosine_similarity = 1 - cosine_distances(concatenated_vectors)
    return cosine_similarity

cosine_similarity = cal_cosine_similarity(vectors)
cosine_similarity_df = pd.DataFrame(cosine_similarity, columns=df['Phrases'], index=df['Phrases'])

print("Cosine Similarity:")
print(cosine_similarity_df)

def fly_execution(user_input,df,vectors):
    input_embedding = word2vec_embedding(user_input)
    cosine_similarity = cal_cosine_similarity([input_embedding] + list(vectors))
    similarity= cosine_similarity[0, 1:]
    closest_idx = np.argmax(similarity)
    closest_mtc = df['Phrases'][closest_idx]
    closest_dis = similarity[closest_idx]

    return {'closest_match': closest_mtc, 'closest_distance': closest_dis}

user_input="hello world!"
print(f"Input: {user_input}")
