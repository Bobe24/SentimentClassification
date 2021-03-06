#!/usr/bin/env python
# -*-coding:utf-8-*-
#  Author: Angela Chapman
#  Date: 8/6/2014
#  Modified by Bobe_24@126.com at 2018/1/10
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Parts 2 and 3 of the tutorial, which cover how to
#  train a model using Word2Vec.
# *************************************** #

import numpy as np
from KaggleWord2VecUtility import KaggleWord2VecUtility


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    # Initialize a counter
    counter = 0.
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float64")
    # Loop through the reviews
    for review in reviews:
        # Print a status message every 1000th review
        if counter % 1000. == 0.:
            print "Review %d of %d" % (counter, len(reviews))
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
                                                         num_features)
        # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs


# ****** Define functions to create average word vectors
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float64")
    nwords = 0.
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews:
        review = review.decode('gbk', 'ignore')
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews

