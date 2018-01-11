# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Bobe_24@126.com
# Function: aggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning

import re
from string import punctuation as punctuation_en
import jieba


class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        # 1. Remove HTML
        # review_text = BeautifulSoup(review).get_text()
        # 2. Remove punctuation and space
        review_text = review
        review_text = re.sub(
            "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。]+".decode(
                "utf8"), "".decode("utf8"),
            review_text)
        review_text = re.sub(r'[{}]+'.format(punctuation_en).decode('utf-8'), "", review_text)
        # 3. Optionally remove stop words (false by default)
        if remove_stopwords:
            review_text = re.sub("[a-zA-Z]", "", review_text)
            review_text = re.sub("[0-9]", "", review_text)
            words = jieba.lcut(review_text)
            path = r"E:\Workspace\DeepLearningMovies\data\stopwords.txt"
            with open(path) as file:
                stopwords = file.readlines()
            stopwords = [stopword.decode('gbk', 'ignore').strip('\n') for stopword in stopwords]
            stops = set(stopwords)
            words = [w for w in words if not w in stops]
        else:
            words = jieba.lcut(review_text)
        # 4. Return a list of words
        return words

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences(review, tokenizer, remove_stopwords=False):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words

        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        # raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        review = review.decode('gbk', 'ignore')
        raw_sentences = re.split(tokenizer.decode('utf-8'), review.strip())
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append(KaggleWord2VecUtility.review_to_wordlist(raw_sentence, \
                                                                          remove_stopwords))
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences
