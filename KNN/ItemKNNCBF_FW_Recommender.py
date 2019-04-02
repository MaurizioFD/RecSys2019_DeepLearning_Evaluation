#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/03/19

@author: Anonymous authors
"""

from Base.Recommender_utils import check_matrix
from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
from Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np
import scipy.sparse as sps

from Base.Similarity.Compute_Similarity import Compute_Similarity


class ItemKNNCBF_FW_Recommender(BaseSimilarityMatrixRecommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCBF_FW_Recommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, ICM_list, URM_train):
        super(ItemKNNCBF_FW_Recommender, self).__init__(URM_train)

        self.n_icm = len(ICM_list)

        self.ICM_list = []
        for ICM in ICM_list:
            self.ICM_list.append(ICM.copy())

        # TODO check that each icm have same number of row (user)


    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", **similarity_args):
        self.ICM_weigth_list = []
        key='icm_wX'
        try:
            for i in range (self.n_icm):
                key = 'icm_w{}'.format(int(i))
                self.ICM_weigth_list.append(float(similarity_args[key]))
                similarity_args.pop(key, None)
        except:
            print('{}: No weight for {} in fit function.'.format(self.RECOMMENDER_NAME, key))

        assert len(self.ICM_weigth_list) == len(self.ICM_list), '{}: number weights and number matrixes not equal'.format(self.RECOMMENDER_NAME)

        for i in range (self.n_icm):
            self.ICM_list[i] = self.ICM_list[i] * self.ICM_weigth_list[i]
            self.ICM_list[i].eliminate_zeros()

        self.ICM = sps.hstack(self.ICM_list)

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))


        if feature_weighting == "BM25":
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = okapi_BM_25(self.ICM)

        elif feature_weighting == "TF-IDF":
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = TF_IDF(self.ICM)


        similarity = Compute_Similarity(self.ICM.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)


        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')


