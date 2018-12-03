import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    xDF = pd.read_csv(filepath_or_buffer="train.csv")
    xDF = xDF.drop("id", 1)
    X = xDF.values

    yDF = xDF.pop("label")
    Y = yDF.values

    # TODO: cross validation here
    xTr, xVer, yTr, yVer = train_test_split(X, Y, test_size=0.8)

    # Please disregard code below:

    # with open("train.csv", "r") as fileHandle:
    #     lines = fileHandle.readlines()
    #
    # titles = lines.pop(0).split(",")
    # index = 1
    # samples = {}
    #
    # for line in lines:
    #     element_dict = {}
    #     _, line_list = line.split(",", 1)
    #     line_list = line_list.split(",")
    #     for i in range(len(titles) - 1):
    #         arg = line_list[i]
    #         element_dict[titles[i]] = arg
    #     samples[index] = element_dict
    #     index += 1
    #
    # # splitting data: 80% training, 20% val
    # # TODO: cross validation here
    # val_size = int(0.2 * index)
    # val_keys = np.random.choice(list(samples.keys()), val_size, replace=False)
    #
    # val_set = {}
    # for k in val_keys:
    #     val_set[k] = samples.pop(k)
    #
    # xTr = np.zeros((len(titles), len(samples.keys())))
    # xVal = np.zeros((len(titles), len(val_keys)))