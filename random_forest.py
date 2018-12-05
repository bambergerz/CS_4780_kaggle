from sklearn.ensemble import RandomForestClassifier
import gensim
import nltk
from nltk.corpus import brown
import os
import pickle


### Random Forest ###

def generate_rf_classifiers(xTr, yTr):
    n = 10
    print("Generating classifier")
    clf = RandomForestClassifier(n_estimators=n)
    print("Fitting classifier")
    clf.fit(xTr, yTr)
    print("Done fitting random forest model. Saving model")
    s = pickle.dumps(clf)
    with open("rf_" + str(n) + ".pickle", 'wb') as fileHandle:
        fileHandle.write(s)
    print("All done!!!")
    return clf
