import gensim
from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format('C:\\Users\\stuti\\OneDrive\\Documents\\DE Galytix assignment\\DE-Galytix-\\GoogleNews-vectors-negative300.bin', binary=True, limit=1000000)
wv.save_word2vec_format('vectors.csv')

