from sklearn.ensemble import RandomForestClassifier
import gensim
import nltk
from nltk.corpus import brown
import os
import pickle


### Random Forest ###

def generate_svm_classifiers(xTr, yTr):
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    pickle.dump(clf, "randomForest.model")
    print("All done!!!")
    return clf
