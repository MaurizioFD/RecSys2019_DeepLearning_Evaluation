#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import time, sys
import scipy.sparse as sps


class Compute_Similarity_Euclidean:


    def __init__(self, dataMatrix, topK=100, shrink = 0, normalize=False, normalize_avg_row=False,
                 similarity_from_distance_mode ="lin", row_weights = None, **args):
        """
        Computes the euclidean similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param normalize
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param similarity_from_distance_mode:       "exp"   euclidean_similarity = 1/(e ^ euclidean_distance)
                                                    "lin"        euclidean_similarity = 1/(1 + euclidean_distance)
                                                    "log"        euclidean_similarity = 1/(1 + euclidean_distance)
        :param args:                accepts other parameters not needed by the current object

        """

        super(Compute_Similarity_Euclidean, self).__init__()

        self.shrink = shrink
        self.normalize = normalize
        self.normalize_avg_row = normalize_avg_row

        self.n_rows, self.n_columns = dataMatrix.shape
        self.TopK = min(topK, self.n_columns)

        self.dataMatrix = dataMatrix.copy()

        self.similarity_is_exp = False
        self.similarity_is_lin = False
        self.similarity_is_log = False

        if similarity_from_distance_mode == "exp":
            self.similarity_is_exp = True
        elif similarity_from_distance_mode == "lin":
            self.similarity_is_lin = True
        elif similarity_from_distance_mode == "log":
            self.similarity_is_log = True
        else:
            raise ValueError("Compute_Similarity_Euclidean: value for parameter 'mode' not recognized."
                             " Allowed values are: 'exp', 'lin', 'log'."
                             " Passed value was '{}'".format(similarity_from_distance_mode))



        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Compute_Similarity_Euclidean: provided row_weights and dataMatrix have different number of rows."
                                 "row_weights has {} rows, dataMatrix has {}.".format(len(row_weights), dataMatrix.shape[0]))

            self.use_row_weights = True
            self.row_weights = row_weights.copy()
            self.row_weights_diag = sps.diags(self.row_weights)

            self.dataMatrix_weighted = self.dataMatrix.T.dot(self.row_weights_diag).T








    def compute_similarity(self, start_col=None, end_col=None, block_size = 100):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """

        values = []
        rows = []
        cols = []

        start_time = time.time()
        start_time_print_batch = start_time
        processedItems = 0


        #self.dataMatrix = self.dataMatrix.toarray()

        start_col_local = 0
        end_col_local = self.n_columns

        if start_col is not None and start_col>0 and start_col<self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col>start_col_local and end_col<self.n_columns:
            end_col_local = end_col

        # Compute sum of squared values
        item_distance_initial = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()
        sumOfSquared = np.sqrt(item_distance_initial)

        start_col_block = start_col_local

        this_block_size = 0

        # Compute all similarities for each item using vectorization
        while start_col_block < end_col_local:

            # Add previous block size
            processedItems += this_block_size

            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block-start_col_block

            if time.time() - start_time_print_batch >= 30 or end_col_block==end_col_local:
                columnPerSec = processedItems / (time.time() - start_time + 1e-9)

                print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / (end_col_local - start_col_local) * 100, columnPerSec, (time.time() - start_time)/ 60))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()


            # All data points for a given item
            item_data = self.dataMatrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()

            # If only 1 feature avoid last dimension to disappear
            if item_data.ndim == 1:
                item_data = np.atleast_2d(item_data)

            if self.use_row_weights:
                this_block_weights = self.dataMatrix_weighted.T.dot(item_data)

            else:
                # Compute item similarities
                this_block_weights = self.dataMatrix.T.dot(item_data)



            for col_index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_column_weights = this_block_weights
                else:
                    this_column_weights = this_block_weights[:,col_index_in_block]


                columnIndex = col_index_in_block + start_col_block

                # item_data = self.dataMatrix[:,columnIndex]

                # (a-b)^2 = a^2 + b^2 - 2ab
                item_distance = item_distance_initial.copy()
                item_distance += item_distance_initial[columnIndex]

                # item_distance -= 2*item_data.T.dot(self.dataMatrix).toarray().ravel()
                item_distance -= 2 * this_column_weights

                item_distance[columnIndex] = 0.0


                if self.use_row_weights:
                    item_distance = np.multiply(item_distance, self.row_weights)


                if self.normalize:
                    item_distance /=  sumOfSquared[columnIndex] * sumOfSquared

                if self.normalize_avg_row:
                    item_distance /= self.n_rows

                item_distance = np.sqrt(item_distance)

                if self.similarity_is_exp:
                    item_similarity = 1/(np.exp(item_distance) + self.shrink + 1e-9)

                elif self.similarity_is_lin:
                    item_similarity = 1/(item_distance + self.shrink + 1e-9)

                elif self.similarity_is_log:
                    item_similarity = 1/(np.log(item_distance+1) + self.shrink + 1e-9)

                else:
                    assert False


                item_similarity[columnIndex] = 0.0

                this_column_weights = item_similarity



                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.TopK-1)[0:self.TopK]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_column_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_column_weights[top_k_idx][notZerosMask])
                rows.extend(top_k_idx[notZerosMask])
                cols.extend(np.ones(numNotZeros) * columnIndex)


            start_col_block += block_size


        # End while on columns

        W_sparse = sps.csr_matrix((values, (rows, cols)),
                                  shape=(self.n_columns, self.n_columns),
                                  dtype=np.float32)

        return W_sparse