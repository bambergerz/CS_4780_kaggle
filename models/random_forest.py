from sklearn.ensemble import RandomForestClassifier
import pickle
import os

#####################
### Random Forest ###
#####################


def generate_rf_classifiers(xTr, yTr):
    os.chdir("data")
    n = 10
    print("Generating classifier")
    clf = RandomForestClassifier(n_estimators=n)
    print("Fitting classifier")
    clf.fit(xTr, yTr)
    print("Done fitting random forest model. Saving model")
    s = pickle.dumps(clf)
    with open("rf_" + str(n) + ".pickle", 'wb') as fileHandle:
        print("Writing rf_" + str(n) + ".pickle to " + os.getcwd())
        fileHandle.write(s)
    os.chdir("..")
    print("All done!!!")
    return clf
