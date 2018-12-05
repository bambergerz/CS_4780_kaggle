import os
import nltk
import gensim
from nltk.corpus import brown


# General pre-processing methods
def get_embeddings():
    if "brown.embedding" in os.listdir(os.getcwd()):
        return gensim.models.Word2Vec.load('brown.embedding')
    nltk.download("brown")
    model = gensim.models.Word2Vec(brown.sents())
    model.save("brown.embedding")
    return model


def get_average_trump_embedding(xTr):
    pass