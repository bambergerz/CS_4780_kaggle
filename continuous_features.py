import numpy as np
import datetime
import json
from urllib.request import urlopen


def capitalized_word_counts(tweets):
    """
    
    :param tweets: n x 1 vector in which entry i represents the text of the i'th tweet. i.e., each entry is a string. 
    :return: n x 1 vector in which entry i represents the number of capitalized words in the i'th tweet, normalized to
    be between 0 and 1
    """
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


def number_of_hastags(tweets):
    """
    
    :param tweets: n x 1 vector in which entry i represents the text of the i'th tweet. i.e., each entry is a string. 
    :return: n x 1 vector in which entry i represents the number of hashtags in the i'th tweet. Normalized to be 
    between 0 and 1
    """
    max_hashtags = 0
    vals = np.zeros(tweets.shape[0])
    for i in range(tweets.shape[0]):
        tweet = tweets[i]
        count = tweet.count("#")
        if count > max_hashtags:
            max_hashtags = count
        vals[i] = count
    return vals / max_hashtags


def tweet_length(tweets):
    """
    
    :param tweets: n x 1 vector in which entry i represents the text of the i'th tweet. i.e., each entry is a string. 
    :return: n x 1 vector in which entry i represents the length (in number of words) of the i'th tweet. Normalized 
    to be between 0 and 1. 
    """
    max_len = 0
    vals = np.zeros(tweets.shape[0])
    for i in range(tweets.shape[0]):
        tweet = tweets[i]
        count = len(tweet.split())
        if count > max_len:
            max_len = count
        vals[i] = count
    return vals / max_len


def favorite_count(tweets):
    """
    
    :param tweets: an n x 1 vector in which each entry i represents the favorite count of tweet i
    :return: a normalized n x 1 vector in which each entry i represents the favorite count of tweet i, but such that
    the value is between 0 and 1
    """
    max_entry = np.amax(tweets)
    return tweets / max_entry


def retweet_count(tweets):
    """
    
    :param tweets: an n x 1 vector in which each entry i represents the retweet count of tweet i
    :return: a normalized n x 1 vector in which each entry i represents the retweet count of tweet i, but such that
    the value is between 0 and 1
    """
    max_entry = np.amax(tweets)
    return tweets / max_entry


def tweet_id(tweets):
    """
    
    :param tweets: an n x 1 vector in which each entry i is the int representing the ID of tweet indexed i
    :return: a normalized n x 1 vector in which each entry i represents the normalized ID of tweet indexed i, but also
    such that the value is between 0 and 1.
    """
    max_entry = np.amax(tweets)
    return tweets / max_entry


def tweet_date(tweets):
    """
    
    :param tweets: an n x 1 vector in which each entry i is a string of the form {month}/{day}/{year} 
    representing the date in which tweet i was released. 
    :return: a n x 1 vector in which each entry i represents the numerical time in which tweet indexed i was
    tweeted. 
    """
    vals = np.zeros(tweets.shape[0])
    for i in range(tweets.shape[0]):
        tweet = tweets[i]
        month, day, year = tweet.split("/")
        time = datetime.datetime(year=int(year),
                                 month=int(month),
                                 day=int(day))
        val = time.timestamp()
        vals[i] = val
    return vals


def tweet_time(tweets):
    """

    :param tweets: an n x 1 vector in which each entry i is a string of the form {military hour}:{minute} 
    representing the time at which tweet i was released. 
    :return: a  n x 1 vector in which each entry i represents the time in which tweet indexed i was tweeted. 
    """
    vals = np.zeros(tweets.shape[0])
    for i in range(tweets.shape[0]):
        tweet = tweets[i]
        hour, minute = tweet.split(":")
        val = 60 * int(hour) + int(minute)
        vals[i] = val
    return vals


def tweet_date_time(tweets):
    """
    
    :param tweet: an n x 1 vector in which each entry i is a string of the form 
                        {month}/{day}/{year} {military hour}:{minute}
                represents the time in which tweet i was released
    :return: date and time vectors as described in the outputs of tweet_time and tweet_date
    """
    dates = np.zeros(tweets.shape[0])
    times = np.zeros(tweets.shape[0])
    for i in range(tweets.shape[0]):
        tweet = tweets[i]
        date, time = tweet.split()
        dates[i] = date
        times[i] = time
    return tweet_date(dates), tweet_time(times)


def tweet_loc(lat, long):
    url = "http://maps.googleapis.com/maps/api/geocode/json?"
    url += "latlng=%s,%s&sensor=false" % (lat, long)
    v = urlopen(url).read()
    j = json.loads(v)
    print(j)


if __name__ == "__main__":
    lat_1 = 40.77010669
    lat_2 = 40.77737697
    long_1 = -73.88530464
    long_2 = -73.88530464
    tweet_loc(lat_1, long_1)
