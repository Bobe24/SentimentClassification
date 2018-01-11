# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Bobe_24@126.com
# Function: main function, call different method

from sklearn import metrics
import xgboost as xgb
import logging
from gensim.models import Word2Vec
from KaggleWord2VecUtility import KaggleWord2VecUtility
from Word2Vec_AverageVectors import getAvgFeatureVecs, getCleanReviews
from Word2Vec_BagOfCentroids import get_centroids
from Tfidf import get_tfidf_vectors
from BagOfWords import get_bow_vectors

# Set values for various parameters
num_features = 300  # Word vector dimensionality
min_word_count = 10  # Minimum word count
num_workers = 2  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-2  # Downsample setting for frequent words


def embedding():
    # Read data from files
    train, test, unlabeled_train = getData()
    # set the punctuation
    tokenizer = r"[.。!！?？;；]"
    # ****** Split the labeled and unlabeled training sets into clean sentences
    sentences = []  # Initialize an empty list of sentences
    print "Parsing sentences from training set"
    for review in train:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
    print "Parsing sentences from unlabeled set"
    for review in unlabeled_train:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
    # ****** Set parameters and train the word2vec model
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                        level=logging.INFO)
    # Initialize and train the model (this will take some time)
    print "Training Word2Vec model..."
    model = Word2Vec(sentences, workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, sample=downsampling, seed=1)
    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)
    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_10minwords_10context"
    model.save(model_name)


def getData():
    # get train data
    posTrainDataPath = r'./data/traindata/pos'
    negTrainDataPath = r'./data/traindata/neg'
    with open(posTrainDataPath) as file:
        train = file.readlines()
    with open(negTrainDataPath) as file:
        train_neg = file.readlines()
    train.extend(train_neg)

    # get test data
    posTestDataPath = r'./data/testdata/pos'
    negTestDataPath = r'./data/testdata/neg'
    with open(posTestDataPath) as file:
        test = file.readlines()
    with open(negTestDataPath) as file:
        test_neg = file.readlines()
    test.extend(test_neg)

    # get unlabeled train data
    unlabeledDataPath = r'./data/unlabeledData.txt'
    with open(unlabeledDataPath) as file:
        unlabeled_train = file.readlines()

    return train, test, unlabeled_train


def classificationModel(trainDataVecs, testDataVecs, method):
    train_label = []
    for item in range(15000):
        train_label.append(1)
    for item in range(15000):
        train_label.append(0)

    test_label = []
    for item in range(5000):
        test_label.append(1)
    for item in range(5000):
        test_label.append(0)

    print "Fitting a XGBoost to labeled training data..."
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(trainDataVecs, train_label)
    # Test & extract results
    result = xgb_model.predict(testDataVecs)
    result_prob = xgb_model.predict_proba(testDataVecs)[:, 1]

    auc = metrics.roc_auc_score(test_label, result_prob)
    f1_score = metrics.f1_score(test_label, result)
    accuracy = metrics.accuracy_score(test_label, result)
    confusion_matrix = metrics.confusion_matrix(test_label, result)
    print ":::: the f1_score of %s-model is %5.6f " % (method, f1_score)
    print ":::: the accuracy of %s-model is %5.6f " % (method, accuracy)
    print ":::: the auc of %s-model is %5.6f " % (method, auc)
    print ":::: the confusionMat of %s-model is " % (method), confusion_matrix


if __name__ == '__main__':
    # train word embedding
    embedding()
    model = Word2Vec.load("300features_10minwords_10context")
    train, test, unlabeled_train = getData()

    # Word2Vec_AverageVectors method
    # ****** Create average vectors for the training and test sets
    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)
    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)
    classificationModel(trainDataVecs, testDataVecs, method='Word2Vec_AverageVectors_xgb')

    # Word2vec_Kmeans method
    train_centroids, test_centroids = get_centroids(model, train, test)
    classificationModel(train_centroids, test_centroids, method='Word2Vec_Kmeans_xgb')

    # Tfidf method
    train_data_features, test_data_features = get_tfidf_vectors(train, test)
    classificationModel(train_data_features, test_data_features, method='Tfidf_xgb')

    # BagOfWords method
    train_data_features, test_data_features = get_bow_vectors(train, test)
    classificationModel(train_data_features, test_data_features, method='BagOfWords_xgb')
