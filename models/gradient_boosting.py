from sklearn.ensemble import GradientBoostingClassifier
import gensim
import nltk
from nltk.corpus import brown
import os
import pickle


### GRADIENT BOOSTING ###

def generate_gb_classifiers(xTr, yTr):
    learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
    classifiers = []
    for learning_rate in learning_rates:
        gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate,
        max_features=2, max_depth = 2, random_state = 0)
        gb.fit(X_train_sub, y_train_sub)
        classifiers.append(c)
    #pickle.dump(logisticRegr, "logisticRegression.model") COMMENTED OUT 11:50pm
    print("All done!!!")
    return classifiers
