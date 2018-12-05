# Libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split

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

    xTr = np.matrix((t_word_count,
                     t_num_words,
                     t_num_hashtags,
                     t_tag_scores,
                     t_hashtags,
                     t_is_tweet_url,
                     t_is_favorited_tweet,
                     t_is_trunc_tweet,
                     t_date,
                     t_time,
                     t_id,
                     t_retweet_count,
                     t_favorite_count)).T
    return xTr


if __name__ == "__main__":

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
    data = pd.DataFrame(xTr)
    print(data)
    print("\n")

    ### SVM ###

    #word_embeddings = get_embeddings()
    # models = svm.generate_svm_classifiers(xTr, yTr)
    model = random_forest.generate_rf_classifiers(xTr, yTr)
    #example of pandas to numpy conversion
    #print(X[0])  # 0 is the text column, indexed by column number
    # print(X[:, 0])
    #print(capitalized_word_counts())


