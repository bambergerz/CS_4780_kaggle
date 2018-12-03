import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm

if __name__ == "__main__":

    xDF = pd.read_csv(filepath_or_buffer="train.csv")
    xDF = xDF.drop("id", 1)
    X = xDF.values

    yDF = xDF.pop("label")
    Y = yDF.values

    # TODO: k-fold cross validation here
    xTr, xVer, yTr, yVer = train_test_split(X, Y, test_size=0.8)

    ### SVM ###

    kernel_types = ["linear", "poly", "rbf", "sigmoid"]
    classifiers = []
    for k_type in kernel_types:
        # enable probability to True for bootstrapping
        c = svm.SVC(kernel=k_type, probability=True)

        # TODO: normalize data so that everything is a float.
        # That way, can fit SVM model

        # c.fit(xTr, yTr)
        # classifiers.append(c)
