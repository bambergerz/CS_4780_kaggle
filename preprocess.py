import os
import numpy as np


if __name__ == "__main__":
    with open("train.csv", "r") as fileHandle:
        lines = fileHandle.readlines()

    titles = lines.pop(0).split(",")
    index = 1
    samples = {}

    for line in lines:
        element_dict = {}
        _, line_list = line.split(",", 1)
        line_list = line_list.split(",")
        for i in range(len(titles) - 1):
            arg = line_list[i]
            element_dict[titles[i]] = arg
        samples[index] = element_dict
        index += 1

    # splitting data: 80% training, 20% val
    # TODO: cross validation here
    val_size = int(0.2 * index)
    val_keys = np.random.choice(list(samples.keys()), val_size, replace=False)

    val_set = {}
    for k in val_keys:
        val_set[k] = samples.pop(k)

    print(val_set)
