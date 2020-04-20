#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

import scipy.sparse as sps
import numpy as np


class ItemKNN_CFCBF_Hybrid_Recommender(ItemKNNCBFRecommender):
    """ ItemKNN_CFCBF_Hybrid_Recommender"""

    RECOMMENDER_NAME = "ItemKNN_CFCBF_HybridRecommender"

    def fit(self, ICM_weight = 1.0, **fit_args):

        self.ICM_train = self.ICM_train*ICM_weight
        self.ICM_train = sps.hstack([self.ICM_train, self.URM_train.T], format='csr')

        super(ItemKNN_CFCBF_Hybrid_Recommender, self).fit(**fit_args)


    def _get_cold_item_mask(self):
        return np.logical_and(self._cold_item_CBF_mask, self._cold_item_mask)

