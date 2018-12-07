import numpy as np
import datetime
import json
from urllib.request import urlopen
import re
from zlib import crc32
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def neg_words(tweets):
    # take in text

    vals = np.zeros(len(tweets))
    for i in range(len(tweets)):

        tweet = (tweets[i]).lower()
        list_tweet = tweet.split()
        if ("crazy" in list_tweet or "weak" in list_tweet or "badly" in list_tweet):
            vals[i] = 1
    return vals


def has_quote_mark(tweet):
    vals = np.zeros(tweet.shape[0])
    for i in range(tweet.shape[0]):
        v = tweet[i]
        if '"' in list(v):
            vals[i] = 1
    return vals

def has_exclamation(tweet):
    vals = np.zeros(tweet.shape[0])
    for i in range(tweet.shape[0]):
        v = tweet[i]
        if '!' in list(v):
            vals[i] = 1
    return vals

def sentiment(tweets):
    """
    param tweets: takes in the tweets text field
    return n x 1 vector where the ith entry represents the positivity and negativity of the tweet itself
    """

    vals = np.zeros(tweets.shape[0])

    analyser = SentimentIntensityAnalyzer()
    for i in range(tweets.shape[0]):
        sentence = tweets[i]
        score = analyser.polarity_scores(sentence)
        vals[i] = score["neg"]

    return vals


def is_favorited(favorited_column):
    """
    
    :param favorited_column: n x 1 vector in which entry i represents FASLE or TRUE depdning on if favorited or not
    :return: n x 1 vector in which entry i represents the 1 if favorited, 0 if not  
    """
    
    vals = np.zeros(favorited_column.shape[0])
    for i in range(favorited_column.shape[0]):
        v = favorited_column[i]
        if v == "TRUE":
        	vals[i] = 1
    return vals


def is_trunc(trunc_column):
    """
    
    :param trunc_column: n x 1 vector in which entry i represents FASLE or TRUE depdning on if truncated or not
    :return: n x 1 vector in which entry i represents the 1 if truncated, 0 if not  
    """
    

    vals = np.zeros(trunc_column.shape[0])
    for i in range(trunc_column.shape[0]):
        v = trunc_column[i]
        if v == "TRUE":
        	vals[i] = 1
    return vals

def bytes_to_float(b):
    c = b.encode()
    return float(crc32(c) & 0xffffffff) / 2**32

def is_tagged(tweets):
    """
    
    :param trunc_column: n x 1 vector in which entry i represents the text content of the tweet
    :return: n x 1 vector in which entry i represents the score of the handles (0 if no handle in tweet )
    """

    vals = np.zeros(tweets.shape[0])
    for i in range(tweets.shape[0]):
        elem = tweets[i]
        match = re.findall(r"(?<=@)\w+", elem)
        if match =="[]":
        	vals[i] = 0
        else:
        	#to_hash = "_".join(match)
        	#hashed_num = bytes_to_float(to_hash)
        	#vals[i] = hashed_num
            vals[i] = 1
    return vals

def has_hashtag(tweets):
    """
    
    :param trunc_column: n x 1 vector in which entry i represents the text content of the tweet
    :return: n x 1 vector in which entry i represents the score of the hash tag in the tweet (0 if no handle in tweet )
    """

    vals = np.zeros(tweets.shape[0])
    for i in range(tweets.shape[0]):
        elem = tweets[i]
        match = re.findall(r"(?<=#)\w+", elem)
        if match =="[]":
        	vals[i] = 0
        else:
        	to_hash = "_".join(match)
        	hashed_num = bytes_to_float(to_hash)
        	vals[i] = hashed_num
    return vals


def is_URL(tweets):
    """
    
    :param trunc_column: n x 1 vector in which entry i represents the text content of the tweet
    :return: n x 1 vector in which entry i represents the score of the url (0 if no url in tweet )
    """

    vals = np.zeros(tweets.shape[0])
    for i in range(tweets.shape[0]):
        elem = tweets[i]
        match = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', elem)
        if match =="[]":
        	vals[i] = 0
        else:
        	to_hash = "_".join(match)
        	hashed_num = bytes_to_float(to_hash)
        	vals[i] = hashed_num
    return vals


def tweet_time2(tweets):
    #take in times
    """

    :param tweets: an n x 1 vector in which each entry i is a string of the form {military hour}:{minute}
    representing the time at which tweet i was released.
    :return: a  n x 1 vector in which each entry i represents the time in which tweet indexed i was tweeted.
    """
    vals = np.zeros(len(tweets))
    for i in range(len(tweets)):
        tweet = tweets[i]
        _, time = tweet.split()
        hour, minute = time.split(":")
        val = int(hour)/24
        #scaled = float(val)/1440
        vals[i] = val
    return vals


def is_3rdperson(tweets):
    # takes in text

    vals = np.zeros(len(tweets))
    for i in range(len(tweets)):
        tweet = (tweets[i]).lower()
        list_tweet = tweet.split()
        if ("trump" in list_tweet or "donald" in list_tweet):
            vals[i] = 1
    return vals


def join_or_tomorrow(tweets):
    # take in text
    vals = np.zeros(len(tweets))

    for i in range(len(tweets)):

        tweet = (tweets[i]).lower()
        list_tweet = tweet.split()
        if ("join" in list_tweet or "tomorrow" in list_tweet):
            vals[i] = 1
    return vals


def is_1stperson(tweets):
    # take in text

    vals = np.zeros(len(tweets))

    for i in range(len(tweets)):

        tweet = (tweets[i]).lower()
        list_tweet = tweet.split()
        if ("i" in list_tweet):
            vals[i] = 1
    return vals






