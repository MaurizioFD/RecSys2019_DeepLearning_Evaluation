#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""

from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Base.Recommender_utils import check_matrix

from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sps


class PureSVDRecommender(BaseMatrixFactorizationRecommender):
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "PureSVDRecommender"

    def __init__(self, URM_train, verbose = True):
        super(PureSVDRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, num_factors=100, random_seed = None):

        self._print("Computing SVD decomposition...")

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      #n_iter=5,
                                      random_state = random_seed)

        s_Vt = sps.diags(Sigma)*VT

        self.USER_factors = U
        self.ITEM_factors = s_Vt.T

        self._print("Computing SVD decomposition... Done!")


