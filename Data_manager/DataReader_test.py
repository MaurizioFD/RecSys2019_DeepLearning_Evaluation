#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/03/18

@author: Anonymous authors
"""

import unittest

import numpy as np


class MyTestCase(unittest.TestCase):

    def test_reconcile_mapper_with_removed_tokens(self):

        from data.DataReader import reconcile_mapper_with_removed_tokens

        # Create mapping [token] -> index

        original_mapper = {"a":0, "b":1, "c":2, "d":3, "e":4}

        reconciled_mapper = reconcile_mapper_with_removed_tokens(original_mapper.copy(), [0])
        assert reconciled_mapper == {"b":0, "c":1, "d":2, "e":3}, "reconciled_mapper not matching control"

        reconciled_mapper = reconcile_mapper_with_removed_tokens(original_mapper.copy(), [4])
        assert reconciled_mapper == {"a":0, "b":1, "c":2, "d":3}, "reconciled_mapper not matching control"

        reconciled_mapper = reconcile_mapper_with_removed_tokens(original_mapper.copy(), [0,2])
        assert reconciled_mapper == {"b":0, "d":1, "e":2}, "reconciled_mapper not matching control"

        reconciled_mapper = reconcile_mapper_with_removed_tokens(original_mapper.copy(), [0,1,2,3,4])
        assert reconciled_mapper == {}, "reconciled_mapper not matching control"




    def test_split_big_CSR_in_columns(self):

        import scipy.sparse as sps
        from Base.Recommender_utils import areURMequals
        from data.DataReader import split_big_CSR_in_columns

        sparse_matrix = sps.random(50, 12, density=0.1, format='csr')

        split_list = split_big_CSR_in_columns(sparse_matrix.copy(), num_split=2)
        split_rebuilt = sps.hstack(split_list)

        assert np.allclose(sparse_matrix.toarray(), split_rebuilt.toarray()), "split_rebuilt not matching sparse_matrix"



        split_list = split_big_CSR_in_columns(sparse_matrix.copy(), num_split=3)
        split_rebuilt = sps.hstack(split_list)

        assert np.allclose(sparse_matrix.toarray(), split_rebuilt.toarray()), "split_rebuilt not matching sparse_matrix"


        split_list = split_big_CSR_in_columns(sparse_matrix.copy(), num_split=5)
        split_rebuilt = sps.hstack(split_list)

        assert np.allclose(sparse_matrix.toarray(), split_rebuilt.toarray()), "split_rebuilt not matching sparse_matrix"


        split_list = split_big_CSR_in_columns(sparse_matrix.copy(), num_split=12)
        split_rebuilt = sps.hstack(split_list)

        assert np.allclose(sparse_matrix.toarray(), split_rebuilt.toarray()), "split_rebuilt not matching sparse_matrix"




if __name__ == '__main__':


    unittest.main()
