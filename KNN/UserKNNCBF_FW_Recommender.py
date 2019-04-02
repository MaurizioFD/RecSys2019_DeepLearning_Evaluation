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


class UserKNNCBF_FW_Recommender(BaseSimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCBF_FW_Recommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, UCM_list, URM_train):
        super(UserKNNCBF_FW_Recommender, self).__init__(URM_train)

        self.n_ucm = len(UCM_list)

        self.UCM_list = []
        for UCM in UCM_list:
            self.UCM_list.append(UCM.copy())

        # TODO check that each ucm have same number of row (user)
        self._compute_item_score = self._compute_score_user_based


    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", **similarity_args):
        self.UCM_weigth_list = []
        key='ucm_wX'
        try:
            for i in range (self.n_ucm):
                key = 'ucm_w{}'.format(int(i))
                self.UCM_weigth_list.append(float(similarity_args[key]))
                similarity_args.pop(key, None)
        except:
            print('{}: No weight for {} in fit function.'.format(self.RECOMMENDER_NAME, key))

        assert len(self.UCM_weigth_list) == len(self.UCM_list), '{}: number weights and number matrixes not equal'.format(self.RECOMMENDER_NAME)

        for i in range (self.n_ucm):
            self.UCM_list[i] = self.UCM_list[i] * self.UCM_weigth_list[i]
            self.UCM_list[i].eliminate_zeros()

        self.UCM = sps.hstack(self.UCM_list)

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))


        if feature_weighting == "BM25":
            self.UCM = self.UCM.astype(np.float32)
            self.UCM = okapi_BM_25(self.UCM)

        elif feature_weighting == "TF-IDF":
            self.UCM = self.UCM.astype(np.float32)
            self.UCM = TF_IDF(self.UCM)


        similarity = Compute_Similarity(self.UCM.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
