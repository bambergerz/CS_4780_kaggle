from sklearn import svm
import gensim
import nltk
from nltk.corpus import brown
import os
import pickle


### SVM ###

def generate_svm_classifiers(xTr, yTr):
    kernel_types = ["linear", "poly", "rbf", "sigmoid"]
    classifiers = []
    for k_type in kernel_types:
        print("training " + k_type + " SVM model...")
        # enable probability to True for bootstrapping
        c = svm.SVC(kernel=k_type, probability=True)
        c.fit(xTr, yTr)
        classifiers.append(c)
        #pickle.dump(c, k_type + "_svm.model")  COMMENTED OUT 11:31
        print("Done!\n")
    print("All done!!!")
    return classifiers
