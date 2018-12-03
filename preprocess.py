# Libraries
import os
import numpy as np
import pandas as pd
import nltk
import gensim
from nltk.corpus import brown
from sklearn.model_selection import train_test_split

# Local imports
import svm


# General pre-processing methods
def get_embeddings():
    if "brown.embedding" in os.listdir(os.getcwd()):
        return gensim.models.Word2Vec.load('brown.embedding')
    nltk.download("brown")
    model = gensim.models.Word2Vec(brown.sents())
    model.save("brown.embedding")
    return model


if __name__ == "__main__":

    xDF = pd.read_csv(filepath_or_buffer="train.csv")
    xDF = xDF.drop("id", 1)
    X = xDF.values

    print("Titles: " + str(xDF.columns.values))

    yDF = xDF.pop("label")
    Y = yDF.values

    # TODO: k-fold cross validation here
    xTr, xVer, yTr, yVer = train_test_split(X, Y, test_size=0.8)

    ### SVM ###

    word_embeddings = get_embeddings()
    svm.svm_eval(xTr, yTr)
