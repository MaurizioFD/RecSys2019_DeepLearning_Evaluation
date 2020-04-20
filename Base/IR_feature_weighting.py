#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/04/18

@author: Maurizio Ferrari Dacrema
"""

import scipy.sparse as sps
import numpy as np


def okapi_BM_25(dataMatrix, K1=1.2, B=0.75):
    """
    Items are assumed to be on rows
    :param dataMatrix:
    :param K1:
    :param B:
    :return:
    """

    assert B>0 and B<1, "okapi_BM_25: B must be in (0,1)"
    assert K1>0,        "okapi_BM_25: K1 must be > 0"

    assert np.all(np.isfinite(dataMatrix.data)), \
        "okapi_BM_25: Data matrix contains {} non finite values".format(np.sum(np.logical_not(np.isfinite(dataMatrix.data))))



    # Weighs each row of a sparse matrix by OkapiBM25 weighting
    # calculate idf per term (user)

    dataMatrix = sps.coo_matrix(dataMatrix)

    N = float(dataMatrix.shape[0])
    idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

    # calculate length_norm per document
    row_sums = np.ravel(dataMatrix.sum(axis=1))

    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    denominator = K1 * length_norm[dataMatrix.row] + dataMatrix.data
    denominator[denominator == 0.0] += 1e-9

    dataMatrix.data = dataMatrix.data * (K1 + 1.0) / denominator * idf[dataMatrix.col]

    return dataMatrix.tocsr()




def TF_IDF(dataMatrix):
    """
    Items are assumed to be on rows
    :param dataMatrix:
    :return:
    """

    assert np.all(np.isfinite(dataMatrix.data)), \
        "TF_IDF: Data matrix contains {} non finite values.".format(np.sum(np.logical_not(np.isfinite(dataMatrix.data))))

    assert np.all(dataMatrix.data >= 0.0),\
        "TF_IDF: Data matrix contains {} negative values, computing the square root is not possible.".format(np.sum(dataMatrix.data < 0.0))

    # TFIDF each row of a sparse amtrix
    dataMatrix = sps.coo_matrix(dataMatrix)
    N = float(dataMatrix.shape[0])

    # calculate IDF
    idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

    # apply TF-IDF adjustment
    dataMatrix.data = np.sqrt(dataMatrix.data) * idf[dataMatrix.col]

    return dataMatrix.tocsr()
