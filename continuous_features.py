import numpy as np


def capitalized_word_counts(tweets):
    max_capitalized_words = 0
    vals = np.zeros(tweets.shape[0])
    for i in range(tweets.shape[0]):
        tweet = tweets[i]
        words = tweet.split()
        capitalized_count = 0
        for word in words:
            if word[0].isUpper():
                capitalized_count += 1
        if capitalized_count > max_capitalized_words:
            max_capitalized_words = capitalized_count
        vals[i] = capitalized_count
    return vals / max_capitalized_words



def number_of_hastags(training_tweets, new_tweet):
    in_tweet = tweet.count("#")


def favorite_count(n):
    pass