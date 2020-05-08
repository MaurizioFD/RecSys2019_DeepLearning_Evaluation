#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""

from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sps
import numpy as np



class PureSVDRecommender(BaseMatrixFactorizationRecommender):
    """ PureSVDRecommender
    Formulation with user latent factors and item latent factors

    As described in Section 3.3.1 of the following article:
    Paolo Cremonesi, Yehuda Koren, and Roberto Turrin. 2010.
    Performance of recommender algorithms on top-n recommendation tasks.
    In Proceedings of the fourth ACM conference on Recommender systems (RecSys ’10).
    Association for Computing Machinery, New York, NY, USA, 39–46.
    DOI:https://doi.org/10.1145/1864708.1864721
    """

    RECOMMENDER_NAME = "PureSVDRecommender"

    def __init__(self, URM_train, verbose = True):
        super(PureSVDRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, num_factors=100, random_seed = None):

        self._print("Computing SVD decomposition...")

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      #n_iter=5,
                                      random_state = random_seed)

        U_s = U * sps.diags(Sigma)

        self.USER_factors = U_s
        self.ITEM_factors = VT.T

        self._print("Computing SVD decomposition... Done!")







def compute_W_sparse_from_item_latent_factors(ITEM_factors, topK = 100):


    n_items, n_factors = ITEM_factors.shape

    block_size = 100

    start_item = 0
    end_item = 0

    values = []
    rows = []
    cols = []


    # Compute all similarities for each item using vectorization
    while start_item < n_items:

        end_item = min(n_items, start_item + block_size)

        this_block_weight = np.dot(ITEM_factors[start_item:end_item, :], ITEM_factors.T)


        for col_index_in_block in range(this_block_weight.shape[0]):

            this_column_weights = this_block_weight[col_index_in_block, :]
            item_original_index = start_item + col_index_in_block

            # Sort indices and select TopK
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            relevant_items_partition = (-this_column_weights).argpartition(topK-1)[0:topK]
            relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
            top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

            # Incrementally build sparse matrix, do not add zeros
            notZerosMask = this_column_weights[top_k_idx] != 0.0
            numNotZeros = np.sum(notZerosMask)

            values.extend(this_column_weights[top_k_idx][notZerosMask])
            rows.extend(top_k_idx[notZerosMask])
            cols.extend(np.ones(numNotZeros) * item_original_index)



        start_item += block_size

    W_sparse = sps.csr_matrix((values, (rows, cols)),
                              shape=(n_items, n_items),
                              dtype=np.float32)

    return W_sparse


from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

class PureSVDItemRecommender(BaseItemSimilarityMatrixRecommender):
    """ PureSVDItemRecommender
    Formulation with the item-item similarity

    As described in Section 3.3.1 of the following article:
    Paolo Cremonesi, Yehuda Koren, and Roberto Turrin. 2010.
    Performance of recommender algorithms on top-n recommendation tasks.
    In Proceedings of the fourth ACM conference on Recommender systems (RecSys ’10).
    Association for Computing Machinery, New York, NY, USA, 39–46.
    DOI:https://doi.org/10.1145/1864708.1864721
    """

    RECOMMENDER_NAME = "PureSVDItemRecommender"

    def __init__(self, URM_train, verbose = True):
        super(PureSVDItemRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, num_factors=100, topK = None, random_seed = None):

        self._print("Computing SVD decomposition...")

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      #n_iter=5,
                                      random_state = random_seed)

        if topK is None:
            topK = self.n_items

        ITEM_factors = VT.T
        W_sparse = compute_W_sparse_from_item_latent_factors(ITEM_factors.T, topK=topK)

        self.W_sparse = sps.csr_matrix(W_sparse)

        self._print("Computing SVD decomposition... Done!")

