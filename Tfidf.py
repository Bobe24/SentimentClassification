# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Bobe_24@126.com
# Function: using tfidf method to create text vectors


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from KaggleWord2VecUtility import KaggleWord2VecUtility
import numpy as np


def get_tfidf_vectors(train, test):
    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []
    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list
    print "Cleaning and parsing the training set movie reviews...\n"
    for i in xrange(0, len(train)):
        train[i] = train[i].decode('gbk', 'ignore')
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train[i], True)))
    # ****** Create a bag of words from the training set
    print "Creating the weight of words using tf-idf...\n"
    # Initialize the "TfidfVectorizer" object
    vectorizer = TfidfVectorizer(analyzer="word", \
                                 tokenizer=None, \
                                 preprocessor=None, \
                                 stop_words=None, \
                                 max_features=5000)
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    # Numpy arrays are easy to work with, so convert the result to an array
    np.asarray(train_data_features)

    # Create an empty list and append the clean reviews one by one
    clean_test_reviews = []
    print "Cleaning and parsing the test set movie reviews...\n"
    for i in xrange(0, len(test)):
        test[i] = test[i].decode('gbk', 'ignore')
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test[i], True)))
    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)

    return train_data_features, test_data_features
