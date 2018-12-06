from sklearn import svm
import gensim
import nltk
from nltk.corpus import brown
import os
import pickle


### Logistic Regression ###

def generate_svm_classifiers(xTr, yTr):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(xTr, yTr)
    pickle.dump(logisticRegr, "logisticRegression.model")
    print("All done!!!")
    return logisticRegr
