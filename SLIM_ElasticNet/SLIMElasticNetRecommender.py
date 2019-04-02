#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anonymous authors
"""


import numpy as np
import scipy.sparse as sps
from Base.Recommender_utils import check_matrix
from sklearn.linear_model import ElasticNet

from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
import time, sys



class SLIMElasticNetRecommender(BaseSimilarityMatrixRecommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    RECOMMENDER_NAME = "SLIMElasticNetRecommender"

    def __init__(self, URM_train):
        super(SLIMElasticNetRecommender, self).__init__(URM_train)


    def fit(self, l1_ratio=0.1, alpha = 1.0, positive_only=True, topK = 100,
            verbose = True):

        assert l1_ratio>= 0 and l1_ratio<=1, "{}: l1_ratio must be between 0 and 1, provided value was {}".format(self.RECOMMENDER_NAME, l1_ratio)

        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)


        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        n_items = URM_train.shape[1]


        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0


        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):

            # get the target column
            y = URM_train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0



            # fit one ElasticNet model per column
            self.model.fit(URM_train, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            # nonzero_model_coef_index = self.model.coef_.nonzero()[0]
            # nonzero_model_coef_value = self.model.coef_[nonzero_model_coef_index]

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data


            local_topK = min(len(nonzero_model_coef_value)-1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]


            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1


            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup


            if verbose and (time.time() - start_time_printBatch > 300 or currentItem == n_items-1):
                print("{}: Processed {} ( {:.2f}% ) in {:.2f} minutes. Items per second: {:.0f}".format(
                    self.RECOMMENDER_NAME,
                    currentItem+1,
                    100.0* float(currentItem+1)/n_items,
                    (time.time()-start_time)/60,
                    float(currentItem)/(time.time()-start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()


        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(n_items, n_items), dtype=np.float32)






import multiprocessing
from multiprocessing import Pool
from functools import partial
import threading

class MultiThreadSLIM_ElasticNet(SLIMElasticNetRecommender, BaseSimilarityMatrixRecommender):

    def __init__(self, URM_train):

        super(MultiThreadSLIM_ElasticNet, self).__init__(URM_train)


    def _partial_fit(self, currentItem, X, topK):
        model = ElasticNet(alpha=1.0,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        # WARNING: make a copy of X to avoid race conditions on column j
        # TODO: We can probably come up with something better here.
        X_j = X.copy()
        # get the target column
        y = X_j[:, currentItem].toarray()
        # set the j-th column of X to zero
        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0
        # fit one ElasticNet model per column
        model.fit(X_j, y)
        # self.model.coef_ contains the coefficient of the ElasticNet model
        # let's keep only the non-zero values
        #nnz_idx = model.coef_ > 0.0

        relevant_items_partition = (-model.coef_).argpartition(topK)[0:topK]
        relevant_items_partition_sorting = np.argsort(-model.coef_[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        notZerosMask = model.coef_[ranking] > 0.0
        ranking = ranking[notZerosMask]

        values = model.coef_[ranking]
        rows = ranking
        cols = [currentItem] * len(ranking)

        #
        # values = model.coef_[nnz_idx]
        # rows = np.arange(X.shape[1])[nnz_idx]
        # cols = np.ones(nnz_idx.sum()) * currentItem
        #
        return values, rows, cols

    def fit(self,l1_ratio=0.1,
                 positive_only=True,
                 topK = 100,
                 workers=multiprocessing.cpu_count()):

        assert l1_ratio>= 0 and l1_ratio<=1, "SLIM_ElasticNet: l1_ratio must be between 0 and 1, provided value was {}".format(l1_ratio)

        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        self.workers = workers




        self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
        n_items = self.URM_train.shape[1]
        # fit item's factors in parallel
        
        #oggetto riferito alla funzione nel quale predefinisco parte dell'input
        _pfit = partial(self._partial_fit, X=self.URM_train, topK=self.topK)
        
        #creo un pool con un certo numero di processi
        pool = Pool(processes=self.workers)
        
        #avvio il pool passando la funzione (con la parte fissa dell'input) 
        #e il rimanente parametro, variabile
        res = pool.map(_pfit, np.arange(n_items))

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    #
    # def fitThreading(self, X):
    #
    #     self.dataset = X
    #     X = check_matrix(X, 'csc', dtype=np.float32)
    #
    #
    #     numThreads = 1
    #
    #
    #     n_items = X.shape[1]
    #
    #     passo = int(n_items/numThreads)
    #     threadList = []
    #
    #
    #
    #     for numThread in range(numThreads):
    #
    #         if numThread == numThreads-1:
    #             end_col = n_items
    #         else:
    #             end_col = passo*(numThread +1)
    #
    #         if numThread == 0:
    #             start_col = 0
    #         else:
    #             start_col = passo*numThread +1
    #
    #         newThread = _myThread(numThread, X, start_col, end_col)
    #         newThread.start()
    #
    #         threadList.append(newThread)
    #
    #
    #     values, rows, cols = [], [], []
    #
    #     for numThread in range(len(threadList)):
    #
    #         threadList[numThread].join
    #
    #         new_values, new_rows, new_cols = threadList[numThread].get_components()
    #
    #         values.extend(new_values)
    #         rows.extend(new_rows)
    #         cols.extend(new_cols)
    #
    #     self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
    #
#
#
#
# class _myThread(threading.Thread):
#
#     def __init__(self, numThread, X, start_col, end_col):
#         super(_myThread, self).__init__()
#
#         l1_penalty=0.1
#         l2_penalty=0.1
#         positive_only=True,
#         l1_ratio = l1_penalty / (l1_penalty + l2_penalty)
#
#         self.numThread = numThread
#         self.start_col = start_col
#         self.end_col = end_col
#         self.X = X.copy()
#         self.model = ElasticNet(alpha=1.0,
#                                l1_ratio=l1_ratio,
#                                positive=positive_only,
#                                fit_intercept=False,
#                                copy_X=False)
#
#
#         self.values = []
#         self.rows = []
#         self.cols = []
#
#
#     def run(self):
#
#         range_col = self.end_col-self.start_col
#
#         message_step = int(range_col*0.05)
#
#         for j in range(self.start_col, self.end_col):
#
#             complete = (j-self.start_col)
#
#
#             if ( complete % message_step == 0):
#                 print('Thread: ' + str(self.numThread) + ' - ' + str(int(complete/range_col*100)) + ' % complete')
#
#             # get the target column
#             y = self.X[:, j].toarray()
#             # set the j-th column of X to zero
#             self.X.data[self.X.indptr[j]:self.X.indptr[j + 1]] = 0.0
#             # fit one ElasticNet model per column
#             self.model.fit(self.X, y)
#             # self.model.coef_ contains the coefficient of the ElasticNet model
#             # let's keep only the non-zero values
#             nnz_idx = self.model.coef_ > 0.0
#             self.values.extend(self.model.coef_[nnz_idx])
#             self.rows.extend(np.arange(self.X.shape[1])[nnz_idx])
#             self.cols.extend(np.ones(nnz_idx.sum()) * j)
#
#         print('Thread: ' + str(self.numThread) + ' - terminated')
#
#     def get_components (self):
#         return self.values, self.rows, self.cols