#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/01/18

@author: Maurizio Ferrari Dacrema
"""


import re
from nltk.stem import PorterStemmer

import nltk
nltk.download('stopwords')


from nltk.corpus import stopwords


def tagFilter(originalTag):

    # Remove non alphabetical character and split on spaces
    processedTag = re.sub("[^a-zA-Z0-9]", " ", originalTag)
    processedTag = re.sub(" +", " ", processedTag)

    processedTag = processedTag.split(" ")

    stopwords_set = set(stopwords.words('english'))

    result = []

    for tag in processedTag:

        if tag not in stopwords_set:
            result.append(tag)

    return result




def tagFilterAndStemming(originalTag):

    # Remove non alphabetical character and split on spaces
    processedTag = re.sub("[^a-zA-Z0-9]", " ", originalTag)
    processedTag = re.sub(" +", " ", processedTag)

    processedTag = processedTag.split(" ")

    stopwords_set = set(stopwords.words('english'))

    stemmer = PorterStemmer()

    result = []

    for tag in processedTag:

        tag_stemmed = stemmer.stem(tag)

        if tag_stemmed not in stopwords_set:
            result.append(tag_stemmed)

    return result