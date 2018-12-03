from sklearn import svm
import gensim
import nltk
from nltk.corpus import brown
import os


### SVM ###

def svm_eval(xTr, yTr):
    kernel_types = ["linear", "poly", "rbf", "sigmoid"]
    classifiers = []
    for k_type in kernel_types:
        # enable probability to True for bootstrapping
        c = svm.SVC(kernel=k_type, probability=True)

        # TODO: normalize data so that everything is a float.
        # That way, can fit SVM model

        c.fit(xTr, yTr)
        classifiers.append(c)



