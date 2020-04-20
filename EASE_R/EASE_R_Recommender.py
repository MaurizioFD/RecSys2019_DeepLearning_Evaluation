#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.Recommender_utils import similarityMatrixTopK
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from sklearn.preprocessing import normalize
import numpy as np
import time
import scipy.sparse as sps

from Base.Similarity.Compute_Similarity import Compute_Similarity


class EASE_R_Recommender(BaseItemSimilarityMatrixRecommender):
    """ EASE_R_Recommender

        https://arxiv.org/pdf/1905.03375.pdf

     @article{steck2019embarrassingly,
      title={Embarrassingly Shallow Autoencoders for Sparse Data},
      author={Steck, Harald},
      journal={arXiv preprint arXiv:1905.03375},
      year={2019}
    }

    """

    RECOMMENDER_NAME = "EASE_R_Recommender"


    def __init__(self, URM_train):
        super(EASE_R_Recommender, self).__init__(URM_train)

    def fit(self, topK=None, l2_norm = 1e3, normalize_matrix = False, verbose = True):

        self.verbose = verbose

        start_time = time.time()
        self._print("Fitting model... ")

        if normalize_matrix:
            # Normalize rows and then columns
            self.URM_train = normalize(self.URM_train, norm='l2', axis=1)
            self.URM_train = normalize(self.URM_train, norm='l2', axis=0)
            self.URM_train = sps.csr_matrix(self.URM_train)


        # Grahm matrix is X X^t, compute dot product
        similarity = Compute_Similarity(self.URM_train, shrink=0, topK=self.URM_train.shape[1], normalize=False, similarity = "cosine")
        grahm_matrix = similarity.compute_similarity().toarray()

        diag_indices = np.diag_indices(grahm_matrix.shape[0])

        grahm_matrix[diag_indices] += l2_norm

        P = np.linalg.inv(grahm_matrix)

        B = P / (-np.diag(P))

        B[diag_indices] = 0.0


        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
        self._print("Fitting model... done in {:.2f} {}".format( new_time_value, new_time_unit))

        if topK is None:
            self.W_sparse = B
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense

        else:
            self.W_sparse = similarityMatrixTopK(B, k = topK, verbose = False)
            self.W_sparse = sps.csr_matrix(self.W_sparse)




    def _compute_score_W_dense(self, user_id_array, items_to_compute = None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse)#.toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse)#.toarray()

        return item_scores





    def load_model(self, folder_path, file_name = None):
        super(EASE_R_Recommender, self).load_model(folder_path, file_name = file_name)

        if not sps.issparse(self.W_sparse):
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense