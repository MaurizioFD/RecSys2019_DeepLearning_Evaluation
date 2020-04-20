#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/09/2018

@author: Maurizio Ferrari Dacrema
"""


import unittest



import scipy.sparse as sps

from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix


def sparse_are_equals(A, B):

    if A.shape != B.shape:
        return False

    return  (A!=B).nnz==0



def random_string():

    import string
    from random import choice, randint

    min_char = 8
    max_char = 12
    allchar = string.ascii_letters + string.digits + string.punctuation
    randomstring = "".join(choice(allchar) for x in range(randint(min_char, max_char)))

    return randomstring



class MyTestCase(unittest.TestCase):


    def test_IncrementalSparseMatrix_add_lists(self):

        n_rows = 100
        n_cols = 200

        randomMatrix = sps.random(n_rows, n_cols, density=0.01, format='coo')

        incrementalMatrix = IncrementalSparseMatrix(n_rows=n_rows, n_cols=n_cols)


        incrementalMatrix.add_data_lists(randomMatrix.row.copy(),
                                         randomMatrix.col.copy(),
                                         randomMatrix.data.copy())


        randomMatrix_incremental = incrementalMatrix.get_SparseMatrix()

        assert sparse_are_equals(randomMatrix, randomMatrix_incremental)







    def test_IncrementalSparseMatrix_add_rows(self):

        import numpy as np

        n_rows = 100
        n_cols = 200

        randomMatrix = sps.random(n_rows, n_cols, density=0.01, format='csr')

        incrementalMatrix = IncrementalSparseMatrix(n_rows=n_rows, n_cols=n_cols)


        for row in range(n_rows):

            row_data = randomMatrix.indices[randomMatrix.indptr[row]:randomMatrix.indptr[row+1]]

            incrementalMatrix.add_single_row(row,
                                             row_data,
                                             5.0)


        randomMatrix.data = np.ones_like(randomMatrix.data)*5.0

        randomMatrix_incremental = incrementalMatrix.get_SparseMatrix()

        assert sparse_are_equals(randomMatrix, randomMatrix_incremental)







if __name__ == '__main__':


    unittest.main()
