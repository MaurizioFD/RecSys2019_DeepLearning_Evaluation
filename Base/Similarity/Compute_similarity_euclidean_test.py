#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

import os
import subprocess
import unittest

import numpy as np
import scipy.sparse as sps

from Base.Recommender_utils import similarityMatrixTopK


def areSparseEquals(Sparse1, Sparse2):

    if(Sparse1.shape != Sparse2.shape):
        return False

    return (Sparse1 - Sparse2).nnz ==0




class MyTestCase(unittest.TestCase):

    def test_euclidean_similarity_integer(self):

        from Base.Similarity.Compute_Similarity_Euclidean import Compute_Similarity_Euclidean
        from scipy.spatial.distance import euclidean

        data_matrix = np.array([[1,1,0,1],[0,1,1,1],[1,0,1,0]])

        n_items = data_matrix.shape[0]

        similarity_object = Compute_Similarity_Euclidean(sps.csr_matrix(data_matrix).T, topK=100, normalize=False, similarity_from_distance_mode="lin")
        W_local = similarity_object.compute_similarity()

        for vector1 in range(n_items):
            for vector2 in range(n_items):

                scipy_distance = euclidean(data_matrix[vector1,:], data_matrix[vector2,:])

                if vector1 == vector2:
                    assert W_local[vector1, vector2] == 0.0, "W_local[{},{}] not matching control".format(vector1, vector2)

                else:
                    local_similarity = 1/W_local[vector1, vector2]

                    assert np.allclose(local_similarity, scipy_distance, atol=1e-4), "W_local[{},{}] not matching control".format(vector1, vector2)




    def test_euclidean_similarity_float(self):

        from Base.Similarity.Compute_Similarity_Euclidean import Compute_Similarity_Euclidean
        from scipy.spatial.distance import euclidean

        data_matrix = np.array([[0.12,0.0,0.87,1.0],
                                [0.28,0.8,1.0,0.69],
                                [0.0,0.37,1.0,0.01]])

        n_items = data_matrix.shape[0]

        similarity_object = Compute_Similarity_Euclidean(sps.csr_matrix(data_matrix).T, topK=100, normalize=False, similarity_from_distance_mode="lin")
        W_local = similarity_object.compute_similarity()

        for vector1 in range(n_items):
            for vector2 in range(n_items):

                scipy_distance = euclidean(data_matrix[vector1,:], data_matrix[vector2,:])

                if vector1 == vector2:
                    assert W_local[vector1, vector2] == 0.0, "W_local[{},{}] not matching control".format(vector1, vector2)

                else:
                    local_similarity = 1/W_local[vector1, vector2]

                    assert np.allclose(local_similarity, scipy_distance, atol=1e-4), "W_local[{},{}] not matching control".format(vector1, vector2)





if __name__ == '__main__':


    unittest.main()