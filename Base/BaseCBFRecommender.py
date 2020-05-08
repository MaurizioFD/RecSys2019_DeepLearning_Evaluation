#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/2017

@author: Maurizio Ferrari Dacrema
"""

from Base.BaseRecommender import BaseRecommender as _BaseRecommender
from Base.Recommender_utils import check_matrix
import numpy as np



class BaseItemCBFRecommender(_BaseRecommender):
    """
    This class refers to a BaseRecommender which uses content features, it provides only one function
    to check if items exist that have no features
    """

    def __init__(self, URM_train, ICM_train, verbose = True):
        super(BaseItemCBFRecommender, self).__init__(URM_train, verbose = verbose)

        assert self.n_items == ICM_train.shape[0], "{}: URM_train has {} items but ICM_train has {}".format(self.RECOMMENDER_NAME, self.n_items, ICM_train.shape[0])
        
        self.ICM_train = check_matrix(ICM_train.copy(), 'csr', dtype=np.float32)
        self.ICM_train.eliminate_zeros()
        
        _, self.n_features = self.ICM_train.shape


        self._cold_item_CBF_mask = np.ediff1d(self.ICM_train.indptr) == 0

        if self._cold_item_CBF_mask.any():
            print("{}: ICM Detected {} ({:.2f} %) items with no features.".format(
                self.RECOMMENDER_NAME, self._cold_item_CBF_mask.sum(), self._cold_item_CBF_mask.sum()/self.n_items*100))






class BaseUserCBFRecommender(_BaseRecommender):
    """
    This class refers to a BaseRecommender which uses content features, it provides only one function
    to check if users exist that have no features
    """

    def __init__(self, URM_train, UCM_train, verbose = True):
        super(BaseUserCBFRecommender, self).__init__(URM_train, verbose = verbose)

        assert self.n_users == UCM_train.shape[0], "{}: URM_train has {} users but UCM_train has {}".format(self.RECOMMENDER_NAME, self.n_items, UCM_train.shape[0])

        self.UCM_train = check_matrix(UCM_train.copy(), 'csr', dtype=np.float32)
        self.UCM_train.eliminate_zeros()

        _, self.n_features = self.UCM_train.shape

        self._cold_user_CBF_mask = np.ediff1d(self.UCM_train.indptr) == 0

        if self._cold_user_CBF_mask.any():
            print("{}: UCM Detected {} ({:.2f} %) cold users.".format(
                self.RECOMMENDER_NAME, self._cold_user_CBF_mask.sum(), self._cold_user_CBF_mask.sum()/self.n_users*100))


