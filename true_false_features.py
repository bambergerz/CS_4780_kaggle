import numpy as np
import datetime
import json
from urllib.request import urlopen
import re
from zlib import crc32


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
	return float(crc32(b) & 0xffffffff) / 2**32

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
        	to_hash = "_".join(match)
        	hashed_num = bytes_to_float(to_hash)
        	vals[i] = hashed_num
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

class MyParser(HTMLParser):
    def __init__(self, output_list=None):
        HTMLParser.__init__(self)
        if output_list is None:
            self.output_list = []
        else:
            self.output_list = output_list
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.output_list.append(dict(attrs).get('href'))






