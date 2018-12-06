from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import numpy as np

#####################
### Random Forest ###
#####################


def generate_rf_classifiers(xTr, yTr):
    cwd = os.getcwd()
    os.chdir(os.path.join("data", "random_forest_models"))
    poss_ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200,
               300, 400, 500, 1000]
    classifiers = []
    for n in poss_ns:

        clf = RandomForestClassifier(n_estimators=n, bootstrap=False)
        clf_bootstrap = RandomForestClassifier(n_estimators=n, bootstrap=True)
        clf.fit(xTr, yTr)
        clf_bootstrap.fit(xTr, yTr)
        classifiers.append(clf)
        classifiers.append(clf_bootstrap)
        s_1 = pickle.dumps(clf)
        s_2 = pickle.dumps(clf_bootstrap)
        with open("rf_" + str(n) + ".pickle", 'wb') as fileHandle:
            fileHandle.write(s_1)
        with open("rf_" + str(n) + "_bootstrap.pickle", "wb") as fileHandle:
            fileHandle.write(s_2)
    os.chdir(cwd)
    return classifiers


def evaluate_classifiers(models, xVer, yVer):
    """

    :param models: a list of models to compare
    :param xVer: the input data of the verification set
    :param yVer: the tags of the verification set
    :return:
    """
    outputs = []
    for model in models:
        output = model.predict(xVer)
        num_samples = yVer.shape[0]
        print("output dims: " + str(output.shape))
        print("yVer dims: " + str(yVer.shape))

        similar = 0
        for i in range(num_samples):
            if output[i] == yVer[i]:
                similar += 1

        accuracy = similar / num_samples
        print("accuracy is: " + str(accuracy) + "\n" * 2)
        outputs.append(accuracy)
    return outputs



