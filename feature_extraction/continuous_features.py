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
            if word[0].isupper():
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


def tweet_loc(lats, longs):
    # TODO: finish this method
    # TODO: find out how to find distance between two points where each point is defined with (lat, long)

    assert lats.shape == longs.shape, "inconsistent shape between lattitude and longitude vectors"
    with open("data.json", "r") as fileHandle:
        d = json.load(fileHandle)

    unique_loc_names     = {}
    unique_state_names   = {}
    unique_county_names  = {}
    unique_country_names = {}

    vals = []

    for i in range(lats.shape[0]):
        lat = lats[i]
        long = longs[i]
        val = [0,0,0,0]

        loc_name = d[(lat, long)]["name"]
        state_name = d[(lat, long)]["admin1"]
        county_name = d[(lat, long)]["admin2"]
        country_name = d[(lat, long)]["cc"]

        #https://stackoverflow.com/questions/26263682/python-add-to-dictionary-loop

        #pseudocode
        # for each of the dicts
        # if loc_name in unique_loc_names:
        #

        #add to val
        val[0] = unique_loc_names[loc_name] #this contains a number
        val[1] = unique_state_names[state_name]
        val[2] = unique_county_names[county_name]
        val[3] = unique_country_names[country_name]
        # unique_country_names.add(country_name)

        vals.append(val)


    loc_names_enums    = set()
    state_name_enums   = set()
    county_name_enums  = set()
    country_name_enums = set()


# Feature ideas:
#   1) Distance from NYC
#   2) Distance from White House
#   3) Distance from Maralago


if __name__ == "__main__":
    lat_1 = 40.77010669
    lat_2 = 40.77737697
    long_1 = -73.88530464
    long_2 = -73.88530464
    # tweet_loc(lat_1, long_1)

