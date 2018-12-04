import numpy as np
import datetime
import json
from urllib.request import urlopen


def capitalized_word_counts(tweets):
    """
    
    :param tweets: n x 1 vector in which entry i represents the text of the i'th tweet. i.e., each entry is a string. 
    :return: n x 1 vector in which entry i represents the number of capitalized words in the i'th tweet, normalized
    """
    vals = np.zeros(tweets.shape[0])
    for i in range(tweets.shape[0]):
        tweet = tweets[i]
        words = tweet.split()
        capitalized_count = 0
        for word in words:
            if word[0].isUpper():
                capitalized_count += 1
        vals[i] = capitalized_count
    return (vals - np.mean(vals)) / np.var(vals)


def number_of_hastags(tweets):
    """
    
    :param tweets: n x 1 vector in which entry i represents the text of the i'th tweet. i.e., each entry is a string. 
    :return: n x 1 vector in which entry i represents the number of hashtags in the i'th tweet, normalized
    """
    vals = np.zeros(tweets.shape[0])
    for i in range(tweets.shape[0]):
        tweet = tweets[i]
        count = tweet.count("#")
        vals[i] = count
    return (vals - np.mean(vals)) / np.var(vals)


def tweet_length(tweets):
    """
    
    :param tweets: n x 1 vector in which entry i represents the text of the i'th tweet. i.e., each entry is a string. 
    :return: n x 1 vector in which entry i represents the length (in number of words) of the i'th tweet, normalized
    """
    vals = np.zeros(tweets.shape[0])
    for i in range(tweets.shape[0]):
        tweet = tweets[i]
        count = len(tweet.split())
        vals[i] = count
    return (vals - np.mean(vals)) / np.var(vals)


def favorite_count(tweets):
    """
    
    :param tweets: an n x 1 vector in which each entry i represents the favorite count of tweet i
    :return: a normalized n x 1 vector in which each entry i represents the favorite count of tweet i, normalized.
    """
    return (tweets - np.mean(tweets)) / np.var(tweets)


def retweet_count(tweets):
    """
    
    :param tweets: an n x 1 vector in which each entry i represents the retweet count of tweet i
    :return: a normalized n x 1 vector in which each entry i represents the retweet count of tweet i, normalized
    """
    return (tweets - np.mean(tweets)) / np.var(tweets)


def tweet_id(tweets):
    """
    
    :param tweets: an n x 1 vector in which each entry i is the int representing the ID of tweet indexed i
    :return: a normalized n x 1 vector in which each entry i represents the normalized ID of tweet indexed i, normalized
    """
    return (tweets - np.mean(tweets)) / np.var(tweets)


def tweet_date(tweets):
    """
    
    :param tweets: an n x 1 array in which each entry i is a string of the form {month}/{day}/{year}
    representing the date in which tweet i was released. 
    :return: a n x 1 vector in which each entry i represents the numerical time in which tweet indexed i was
    tweeted. 
    """
    vals = np.zeros(len(tweets))
    for i in range(len(tweets)):
        tweet = tweets[i]
        month, day, year = tweet.split("/")
        time = datetime.datetime(year=(2000 + int(year)),
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
    vals = np.zeros(len(tweets))
    for i in range(len(tweets)):
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
    dates = []
    times = []
    for i in range(tweets.shape[0]):
        tweet = tweets[i]
        date, time = tweet.split()
        dates.append(date)
        times.append(time)
    return tweet_date(dates), tweet_time(times)


def tweet_loc(lat, long):
    # TODO: finish this method
    # TODO: find out how to find distance between two points where each point is defined with (lat, long)
    url = "http://maps.googleapis.com/maps/api/geocode/json?"
    url += "latlng=%s,%s&sensor=false" % (lat, long)
    v = urlopen(url).read()
    j = json.loads(v)
    print(j)


# Feature ideas:
#   1) Distance from NYC
#   2) Distance from White House
#   3) Distance from Maralago


if __name__ == "__main__":
    lat_1 = 40.77010669
    lat_2 = 40.77737697
    long_1 = -73.88530464
    long_2 = -73.88530464
    tweet_loc(lat_1, long_1)

