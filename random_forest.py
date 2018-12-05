from sklearn.ensemble import RandomForestClassifier
import gensim
import nltk
from nltk.corpus import brown
import os
import pickle


### Random Forest ###

def generate_svm_classifiers(xTr, yTr):
    clf=RandomForestClassifier(n_estimators=10)
    clf.fit(xTr,yTr)
    pickle.dump(clf, "randomForest.model")
    print("All done!!!")
    return clf
