#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender


class ItemKNNCustomSimilarityRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCustomSimilarityRecommender"

    def fit(self, W_sparse, selectTopK = False, topK=100):

        assert W_sparse.shape[0] == W_sparse.shape[1],\
            "ItemKNNCustomSimilarityRecommender: W_sparse matrice is not square. Current shape is {}".format(W_sparse.shape)

        assert self.URM_train.shape[1] == W_sparse.shape[0],\
            "ItemKNNCustomSimilarityRecommender: URM_train and W_sparse matrices are not consistent. " \
            "The number of columns in URM_train must be equal to the rows in W_sparse. " \
            "Current shapes are: URM_train {}, W_sparse {}".format(self.URM_train.shape, W_sparse.shape)

        if selectTopK:
            W_sparse = similarityMatrixTopK(W_sparse, k=topK)

        self.W_sparse = check_matrix(W_sparse, format='csr')