from sklearn.linear_model import LogisticRegression
import gensim
import nltk
from nltk.corpus import brown
import os
import pickle


### Logistic Regression ###

def generate_log_classifiers(xTr, yTr):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(xTr, yTr)
    #pickle.dump(logisticRegr, "logisticRegression.model") COMMENTED OUT 11:50pm
    print("All done!!!")
    return logisticRegr
