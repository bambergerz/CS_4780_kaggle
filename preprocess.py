# Libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from models import svm
from models import logistic_regression

# Local imports
from models import random_forest
from feature_extraction.continuous_features import *
from feature_extraction.true_false_features import *


def get_features(X):
    """

    :param X:
    :return:
    """
    TEXT           = X[:, 0]
    FAVORITED      = X[:, 1]
    FAVORITE_COUNT = X[:, 2]
    REPLY_TO_SN    = X[:, 3]
    CREATED        = X[:, 4]
    TRUNCATED      = X[:, 5]
    REPLY_TO_SID   = X[:, 6]
    ID             = X[:, 7]
    REPLY_TO_UID   = X[:, 8]
    STATUS_SOURCE  = X[:, 9]
    SCREEN_NAME    = X[:, 10]
    RETWEET_COUNT  = X[:, 11]
    IS_RETWEET     = X[:, 12]
    RETWEETED      = X[:, 13]
    LONGITUDE      = X[:, 14]
    LATITUDE       = X[:, 15]
    LABEL          = X[:, 16]
    #print(STATUS_SOURCE)

    t_quote_mark = has_quote_mark(TEXT)
    t_word_count = number_of_hastags(TEXT)
    t_num_words = tweet_length(TEXT)
    t_num_hashtags = number_of_hastags(TEXT)
    t_tag_scores = is_tagged(TEXT)                      # TODO: confirm with Neel
    t_hashtags = has_hashtag(TEXT)                      # TODO: confirm with Neel
    t_is_tweet_url = is_URL(TEXT)                       # TODO: confirm with Neel
    t_is_favorited_tweet = is_favorited(FAVORITED)
    t_is_trunc_tweet = is_trunc(TRUNCATED)
    t_date, t_time = tweet_date_time(CREATED)
    t_id = tweet_id(ID)
    t_retweet_count = retweet_count(RETWEET_COUNT)
    t_favorite_count = favorite_count(FAVORITE_COUNT)
    t_sentiment_score = sentiment(TEXT)
    # new features
    t_tweet_time_2 = tweet_time2(CREATED)
    t_is_third_person = is_3rdperson(TEXT)
    t_join_or_tomorrow = join_or_tomorrow(TEXT)
    t_is_first_person = is_1stperson(TEXT)
    t_neg_words = neg_words(TEXT)
    t_num_exclamation = number_of_exclamation(TEXT)
    t_has_exclamation = has_exclamation(TEXT)

    # t_is_favorited_tweet,      # .56 bad
    # t_is_trunc_tweet,          # .56
    # t_date,                      #.57
    # t_time,                       #.57
    # t_id,                            #.56
    # t_retweet_count,                  #.56
    # t_favorite_count,                   #.55
    # t_sentiment_score  #pretty bad as a feature .57
    # t_tweet_time_2,    #.58 log
    # t_is_third_person,  #.57 log
    # t_join_or_tomorrow, #.6
    # t_is_first_person,  #.55
    # t_neg_words
    # t_num_exclamation,
    # t_has_exclamation
    # t_tag_scores,         #.56 very bad
    xTr = np.matrix((
                        t_quote_mark,   #.63
                        t_word_count,  # pretty good .76
                        t_num_words,  # .69 #capitalized words
                        t_num_hashtags,  # .75
                        t_hashtags,  # .74
                        t_is_tweet_url,  # .81 EXTREMELY GOOD
                     )).T
    return xTr

def get_features_test(X):
    """

    :param X:
    :return:
    """
    TEXT           = X[:, 0]
    FAVORITED      = X[:, 1]
    FAVORITE_COUNT = X[:, 2]
    REPLY_TO_SN    = X[:, 3]
    CREATED        = X[:, 4]
    TRUNCATED      = X[:, 5]
    REPLY_TO_SID   = X[:, 6]
    ID             = X[:, 7]
    REPLY_TO_UID   = X[:, 8]
    SCREEN_NAME    = X[:, 9]
    RETWEET_COUNT  = X[:, 10]
    IS_RETWEET     = X[:, 11]
    RETWEETED      = X[:, 12]
    LONGITUDE      = X[:, 13]
    LATITUDE       = X[:, 14]

    t_quote_mark = has_quote_mark(TEXT)
    t_word_count = capitalized_word_counts(TEXT)
    t_num_words = tweet_length(TEXT)
    t_num_hashtags = number_of_hastags(TEXT)
    t_tag_scores = is_tagged(TEXT)                      # TODO: confirm with Neel
    t_hashtags = has_hashtag(TEXT)                      # TODO: confirm with Neel
    t_is_tweet_url = is_URL(TEXT)                       # TODO: confirm with Neel
    t_is_favorited_tweet = is_favorited(FAVORITED)
    t_is_trunc_tweet = is_trunc(TRUNCATED)
    t_date, t_time = tweet_date_time(CREATED)
    t_id = tweet_id(ID)
    t_retweet_count = retweet_count(RETWEET_COUNT)
    t_favorite_count = favorite_count(FAVORITE_COUNT)
    t_sentiment_score = sentiment(TEXT)
    #new features
    t_tweet_time_2 = tweet_time2(CREATED)
    t_is_third_person = is_3rdperson(TEXT)
    t_join_or_tomorrow = join_or_tomorrow(TEXT)
    t_is_first_person = is_1stperson(TEXT)
    t_neg_words = neg_words(TEXT)
    t_num_exclamation = number_of_exclamation(TEXT)
    t_has_exclamation = has_exclamation(TEXT)
    xTr = np.matrix((
            t_quote_mark,
            t_word_count,  # pretty good .76
            t_num_words,  # .69 #capitalized words
            t_num_hashtags,  # .75
            t_hashtags,  # .74
            t_is_tweet_url,  # .81 EXTREMELY GOOD
                         )).T
    return xTr


def predict_test(models, model_scores):
    cwd = os.getcwd()
    os.chdir("data")
    xDF = pd.read_csv(filepath_or_buffer="test.csv")
    xDF = xDF.drop("id", 1)
    X = xDF.values
    xTe = get_features_test(X)
    i = np.argmax(np.array(model_scores))
    print("i is", i)
    best_model = models[i]
    output = pd.DataFrame(best_model.predict(xTe), columns=['label'])
    with open("predictions.csv", "w") as fileHandle:
        output.to_csv(fileHandle)
    os.chdir(cwd)


def submission_full():
    cwd = os.getcwd()
    os.chdir("data")
    xDF = pd.read_csv(filepath_or_buffer="train.csv")
    xDF = xDF.drop("id", 1)
    X = xDF.values
    yDF = xDF.pop("label")
    Y = yDF.values
    xTr = get_features(X)

    clf_bootstrap = RandomForestClassifier(n_estimators=60, bootstrap=True)
    clf_bootstrap.fit(xTr, Y)

    xDF_test = pd.read_csv(filepath_or_buffer="test.csv")
    xDF_test = xDF_test.drop("id", 1)
    X_test = xDF_test.values
    xTe = get_features_test(X_test)

    output = pd.DataFrame(clf_bootstrap.predict(xTe), columns=['label'])
    with open("full_predictions.csv", "w") as fileHandle:
        output.to_csv(fileHandle)
    os.chdir(cwd)



if __name__ == "__main__":

    cwd = os.getcwd()
    os.chdir("data")
    xDF = pd.read_csv(filepath_or_buffer="train.csv")
    xDF = xDF.drop("id", 1)
    X = xDF.values
    os.chdir("..")

    print("Titles: " + str(xDF.columns.values) + "\n")

    yDF = xDF.pop("label")
    Y = yDF.values

    # TODO: k-fold cross validation here
    xTr, xVer, yTr, yVer = train_test_split(X, Y, test_size=0.8)

    xTr = get_features(xTr)
    xVer = get_features(xVer)


    ## LOGISTIC REGRESSION ##
    #models = [logistic_regression.generate_log_classifiers(xTr, yTr)] #made it a list

    ### SVM ###

    #word_embeddings = get_embeddings()
    models = svm.generate_svm_classifiers(xTr, yTr)

    ### Random Forest ###
    # os.chdir(cwd)
    # models = random_forest.generate_rf_classifiers(xTr, yTr) #UNCOMMENT THIS WHEN ADDING FEATURES
    # models = []
    # cwd = os.getcwd()
    # os.chdir("data")
    # os.chdir("random_forest_models")
    # files = os.listdir(os.getcwd())
    # for file in files:
    #     with open(file, "rb") as fileHandle:
    #         s = fileHandle.read()
    #         model = pickle.loads(s)
    #         models.append(model)
    model_scores = random_forest.evaluate_classifiers(models, xVer, yVer)
    os.chdir(cwd)

    print("model scores are: \n" + str(model_scores))
    print("max accuracy was: " + str(max(model_scores)))

    predict_test(models, model_scores)
    #submission_full()


    #example of pandas to numpy conversion
    #print(X[0])  # 0 is the text column, indexed by column number
    # print(X[:, 0])
    #print(capitalized_word_counts())


