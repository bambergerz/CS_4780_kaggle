# Libraries
import os
import numpy as np
import pandas as pd
import nltk
import gensim
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from continuous_features import *

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

def get_average_trump_embedding(xTr):
    pass


if __name__ == "__main__":

    xDF = pd.read_csv(filepath_or_buffer="train.csv")
    xDF = xDF.drop("id", 1)
    X = xDF.values

    print("Titles: " + str(xDF.columns.values) + "\n")

    yDF = xDF.pop("label")
    Y = yDF.values

    # TODO: k-fold cross validation here
    xTr, xVer, yTr, yVer = train_test_split(X, Y, test_size=0.8)

    ### SVM ###

    #word_embeddings = get_embeddings()
    #svm.svm_eval(xTr, yTr)

    #example of pandas to numpy conversion
    #print(X[0])  # 0 is the text column, indexed by column number
    print(X[:, 0])
    #print(capitalized_word_counts())

    #columns in the csv
    TEXT           = X[:, 0]
    FAVORITED      = X[:, 1]
    FAVORITE_COUNT = X[:, 2]
    REPLY_TO_SN    = X[:, 3]
    CREATED        = X[:, 4]
    TRUNCATED      = X[:, 5]
    REPLY_TO_SID   = X[:, 6]
    ID             = X[:, 7]
    REPLY_TO_UID   = X[:, 8]
    STATUS_SOURCE  = X[:, 9]
    SCREEN_NAME    = X[:, 10]
    RETWEET_COUNT  = X[:, 11]
    IS_RETWEET     = X[:, 12]
    RETWEETED      = X[:, 13]
    LONGITUDE      = X[:, 14]
    LATITUDE       = X[:, 15]
    LABEL          = X[:, 16]
    #print(STATUS_SOURCE)

    """'text' 'favorited' 'favoriteCount' 'replyToSN' 'created' 'truncated'
    'replyToSID' 'id.1' 'replyToUID' 'statusSource' 'screenName'
    'retweetCount' 'isRetweet' 'retweeted' 'longitude' 'latitude' 'label']"""

