#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

import scipy.sparse as sps


class ItemKNN_CFCBF_Hybrid_Recommender(ItemKNNCBFRecommender, BaseSimilarityMatrixRecommender):
    """ ItemKNN_CFCBF_Hybrid_Recommender"""

    RECOMMENDER_NAME = "ItemKNN_CFCBF_HybridRecommender"

    def fit(self, ICM_weight = 1.0, **fit_args):

        self.ICM = self.ICM*ICM_weight
        self.ICM = sps.hstack([self.ICM, self.URM_train.T], format='csr')

        super(ItemKNN_CFCBF_Hybrid_Recommender, self).fit(**fit_args)

