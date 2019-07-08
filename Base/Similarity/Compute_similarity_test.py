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

    def test_cosine_similarity_dense(self):

        from Base.Cython.cosine_similarity import Cosine_Similarity as Cosine_Similarity_Cython
        from Base.cosine_similarity_parallel import Cosine_Similarity_Parallel as Cosine_Similarity_Parallel

        TopK = 0

        data_matrix = np.array([[1,1,0,1],[0,1,1,1],[1,0,1,0]])
        data_matrix = sps.csr_matrix(data_matrix)

        cosine_similarity = Cosine_Similarity_Cython(data_matrix, topK=TopK, normalize = False)
        W_dense_Cython = cosine_similarity.compute_similarity()

        cosine_similarity = Compute_Similarity_Python(data_matrix, topK=TopK, normalize = False)
        W_dense_Python = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Parallel(data_matrix, topK=TopK, normalize = False)
        W_dense_Parallel = cosine_similarity.compute_similarity()


        W_dense_mul = data_matrix.T.dot(data_matrix)
        W_dense_mul[np.arange(W_dense_mul.shape[0]),np.arange(W_dense_mul.shape[0])] = 0.0

        assert np.all(W_dense_Cython == W_dense_mul), "W_dense_Cython not matching control"
        assert np.all(W_dense_Python == W_dense_mul), "W_dense_Python not matching control"
        assert np.all(W_dense_Parallel == W_dense_mul), "W_dense_Parallel not matching control"


    def test_cosine_similarity_dense_row_weighted(self):

        from Base.Cython.cosine_similarity import Cosine_Similarity as Cosine_Similarity_Cython
        from Base.cosine_similarity_parallel import Cosine_Similarity_Parallel as Cosine_Similarity_Parallel

        TopK = 0

        data_matrix = np.array([[1,2,0,1],[0,1,4,1],[3,0,1,0]])
        data_matrix = sps.csr_matrix(data_matrix, dtype=np.float)

        row_weights = [2, 3, 0, 4]

        cosine_similarity = Cosine_Similarity_Cython(data_matrix.T, topK=TopK, normalize = False, row_weights = row_weights)
        W_dense_Cython = cosine_similarity.compute_similarity()

        cosine_similarity = Compute_Similarity_Python(data_matrix.T, topK=TopK, normalize = False, row_weights = row_weights)
        W_dense_Python = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Parallel(data_matrix.T, topK=TopK, normalize = False, row_weights = row_weights)
        W_dense_Parallel = cosine_similarity.compute_similarity()


        W_dense_mul = data_matrix.dot(sps.diags(row_weights)).dot(data_matrix.T).toarray()
        W_dense_mul[np.arange(W_dense_mul.shape[0]),np.arange(W_dense_mul.shape[0])] = 0.0

        assert np.allclose(W_dense_Cython, W_dense_mul, atol=1e-4), "W_dense_Cython not matching control"
        assert np.allclose(W_dense_Python, W_dense_mul, atol=1e-4), "W_dense_Python not matching control"
        assert np.allclose(W_dense_Parallel, W_dense_mul, atol=1e-4), "W_dense_Parallel not matching control"




    def test_cosine_similarity_dense_external_cfr(self):

        from Base.Cython.cosine_similarity import Cosine_Similarity as Cosine_Similarity_Cython
        from Base.cosine_similarity_parallel import Cosine_Similarity_Parallel as Cosine_Similarity_Parallel
        from sklearn.metrics.pairwise import cosine_similarity as Cosine_Similarity_Sklearn


        from scipy.spatial.distance import jaccard as Jaccard_Distance_Scipy


        TopK = 0
        shrink = 0

        data_matrix = np.array([[1,2,0,1],[0,1,4,1],[1,3,1,0]])
        data_matrix = sps.csr_matrix(data_matrix)

        cosine_similarity = Cosine_Similarity_Cython(data_matrix, topK=TopK, normalize = True, shrink=shrink)
        W_dense_Cython = cosine_similarity.compute_similarity()

        cosine_similarity = Compute_Similarity_Python(data_matrix, topK=TopK, normalize = True, shrink=shrink)
        W_dense_Python = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Parallel(data_matrix, topK=TopK, normalize = True, shrink=shrink)
        W_dense_Parallel = cosine_similarity.compute_similarity()


        W_dense_sklearn = Cosine_Similarity_Sklearn(data_matrix.copy().T)
        W_dense_sklearn[np.arange(W_dense_sklearn.shape[0]),np.arange(W_dense_sklearn.shape[0])] = 0.0


        assert np.allclose(W_dense_Cython, W_dense_sklearn, atol=1e-4), "W_dense_Cython Cosine not matching Sklearn control"
        assert np.allclose(W_dense_Python, W_dense_sklearn, atol=1e-4), "W_dense_Python Cosine not matching Sklearn control"
        assert np.allclose(W_dense_Parallel, W_dense_sklearn, atol=1e-4), "W_dense_Parallel Cosine not matching Sklearn control"


        data_matrix = np.array([[1,2,0,1],[0,1,4,1],[1,3,1,0]])
        data_matrix = sps.csr_matrix(data_matrix)


        cosine_similarity = Cosine_Similarity_Cython(data_matrix, topK=TopK, normalize = True, shrink=shrink,
                                                     mode='jaccard')
        W_dense_Cython = cosine_similarity.compute_similarity()

        cosine_similarity = Compute_Similarity_Python(data_matrix, topK=TopK, normalize = True, shrink=shrink,
                                                      mode='jaccard')
        W_dense_Python = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Parallel(data_matrix, topK=TopK, normalize = True, shrink=shrink,
                                                     mode='jaccard')
        W_dense_Parallel = cosine_similarity.compute_similarity()


        W_dense_Scipy = np.zeros_like(W_dense_Python)
        data_matrix.data = np.ones_like(data_matrix.data)
        data_matrix = data_matrix.toarray()

        for row in range(W_dense_Scipy.shape[0]):
            for col in range(W_dense_Scipy.shape[1]):

                if row != col:
                    W_dense_Scipy[row, col] = 1-Jaccard_Distance_Scipy(data_matrix[:,row], data_matrix[:,col])


        assert np.allclose(W_dense_Cython, W_dense_Scipy, atol=1e-4), "W_dense_Cython Jaccard not matching Scipy control"
        assert np.allclose(W_dense_Python, W_dense_Scipy, atol=1e-4), "W_dense_Python Jaccard not matching Scipy control"
        assert np.allclose(W_dense_Parallel, W_dense_Scipy, atol=1e-4), "W_dense_Parallel Jaccard not matching Scipy control"





    def test_cosine_similarity_dense_normalize(self):

        from Base.Cython.cosine_similarity import Cosine_Similarity as Cosine_Similarity_Cython
        from Base.cosine_similarity import Compute_Similarity as Cosine_Similarity_Python
        from Base.cosine_similarity_parallel import Cosine_Similarity_Parallel as Cosine_Similarity_Parallel

        import numpy.matlib

        TopK = 0
        shrink = 5

        data_matrix = np.array([[1,1,0,1],[0,1,1,1],[1,0,1,0]])
        data_matrix = sps.csr_matrix(data_matrix)

        cosine_similarity = Cosine_Similarity_Cython(data_matrix, topK=TopK, normalize = True, shrink=shrink)
        W_dense_Cython = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Python(data_matrix, topK=TopK, normalize = True, shrink=shrink)
        W_dense_Python = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Parallel(data_matrix, topK=TopK, normalize = True, shrink=shrink)
        W_dense_Parallel = cosine_similarity.compute_similarity()


        W_dense_denominator = np.matlib.repmat(data_matrix.power(2).sum(axis=0), data_matrix.shape[1], 1)
        W_dense_denominator = np.sqrt(W_dense_denominator)
        W_dense_denominator = np.multiply(W_dense_denominator, W_dense_denominator.T) + shrink

        W_dense_mul = data_matrix.T.dot(data_matrix)
        W_dense_mul /= W_dense_denominator

        W_dense_mul[np.arange(W_dense_mul.shape[0]),np.arange(W_dense_mul.shape[0])] = 0.0


        assert np.allclose(W_dense_Cython, W_dense_mul, atol=1e-4), "W_dense_Cython not matching control"
        assert np.allclose(W_dense_Python, W_dense_mul, atol=1e-4), "W_dense_Python not matching control"
        assert np.allclose(W_dense_Parallel, W_dense_mul, atol=1e-4), "W_dense_Parallel not matching control"




    def test_cosine_similarity_dense_adjusted(self):

        from Base.Cython.cosine_similarity import Cosine_Similarity as Cosine_Similarity_Cython
        from Base.cosine_similarity import Compute_Similarity as Cosine_Similarity_Python
        from Base.cosine_similarity_parallel import Cosine_Similarity_Parallel as Cosine_Similarity_Parallel

        import numpy.matlib

        TopK = 0
        shrink = 0

        data_matrix = np.array([[1,2,0,1],[0,1,4,1],[1,3,1,0]])
        data_matrix = sps.csr_matrix(data_matrix)

        cosine_similarity = Cosine_Similarity_Cython(data_matrix, topK=TopK, normalize = True,
                                                     shrink=shrink, mode='adjusted')
        W_dense_Cython = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Python(data_matrix, topK=TopK, normalize = True,
                                                     shrink=shrink, mode='adjusted')
        W_dense_Python = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Parallel(data_matrix, topK=TopK, normalize = True,
                                                     shrink=shrink, mode='adjusted')
        W_dense_Parallel = cosine_similarity.compute_similarity()


        data_matrix = data_matrix.toarray().astype(np.float64)
        for row in range(data_matrix.shape[0]):

            nonzeroMask = data_matrix[row,:]>0
            data_matrix[row,:][nonzeroMask] -= np.mean(data_matrix[row,:][nonzeroMask])


        W_dense_denominator = np.matlib.repmat((data_matrix**2).sum(axis=0), data_matrix.shape[1], 1)
        W_dense_denominator = np.sqrt(W_dense_denominator)
        W_dense_denominator = np.multiply(W_dense_denominator, W_dense_denominator.T) + shrink

        W_dense_mul = data_matrix.T.dot(data_matrix)
        W_dense_mul[W_dense_denominator>0] /= W_dense_denominator[W_dense_denominator>0]

        W_dense_mul[np.arange(W_dense_mul.shape[0]),np.arange(W_dense_mul.shape[0])] = 0.0

        assert np.allclose(W_dense_Cython, W_dense_mul, atol=1e-4), "W_dense_Cython not matching control"
        assert np.allclose(W_dense_Python, W_dense_mul, atol=1e-4), "W_dense_Python not matching control"
        assert np.allclose(W_dense_Parallel, W_dense_mul, atol=1e-4), "W_dense_Parallel not matching control"



    def test_cosine_similarity_dense_pearson(self):

        from Base.Cython.cosine_similarity import Cosine_Similarity as Cosine_Similarity_Cython
        from Base.cosine_similarity import Compute_Similarity as Cosine_Similarity_Python
        from Base.cosine_similarity_parallel import Cosine_Similarity_Parallel as Cosine_Similarity_Parallel

        import numpy.matlib

        TopK = 0
        shrink = 0

        data_matrix = np.array([[1,2,0,1],[0,1,4,1],[1,3,1,0]])
        data_matrix = sps.csr_matrix(data_matrix)

        cosine_similarity = Cosine_Similarity_Cython(data_matrix, topK=TopK, normalize = True,
                                                     shrink=shrink, mode='pearson')
        W_dense_Cython = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Python(data_matrix, topK=TopK, normalize = True,
                                                     shrink=shrink, mode='pearson')
        W_dense_Python = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Parallel(data_matrix, topK=TopK, normalize = True,
                                                     shrink=shrink, mode='pearson')
        W_dense_Parallel = cosine_similarity.compute_similarity()


        data_matrix = data_matrix.toarray().astype(np.float64)
        for col in range(data_matrix.shape[1]):

            nonzeroMask = data_matrix[:,col]>0
            data_matrix[:,col][nonzeroMask] -= np.mean(data_matrix[:,col][nonzeroMask])


        W_dense_denominator = np.matlib.repmat((data_matrix**2).sum(axis=0), data_matrix.shape[1], 1)
        W_dense_denominator = np.sqrt(W_dense_denominator)
        W_dense_denominator = np.multiply(W_dense_denominator, W_dense_denominator.T) + shrink

        W_dense_mul = data_matrix.T.dot(data_matrix)
        W_dense_mul[W_dense_denominator>0] /= W_dense_denominator[W_dense_denominator>0]

        W_dense_mul[np.arange(W_dense_mul.shape[0]),np.arange(W_dense_mul.shape[0])] = 0.0

        assert np.allclose(W_dense_Cython, W_dense_mul, atol=1e-4), "W_dense_Cython not matching control"
        assert np.allclose(W_dense_Python, W_dense_mul, atol=1e-4), "W_dense_Python not matching control"
        assert np.allclose(W_dense_Parallel, W_dense_mul, atol=1e-4), "W_dense_Parallel not matching control"



    def test_cosine_similarity_dense_jaccard(self):

        from Base.Cython.cosine_similarity import Cosine_Similarity as Cosine_Similarity_Cython
        from Base.cosine_similarity import Compute_Similarity as Cosine_Similarity_Python
        from Base.cosine_similarity_parallel import Cosine_Similarity_Parallel as Cosine_Similarity_Parallel

        import numpy.matlib

        TopK = 0
        shrink = 0

        data_matrix = np.array([[1,2,0,1],[0,1,4,1],[1,3,1,0]])
        data_matrix = sps.csr_matrix(data_matrix)

        cosine_similarity = Cosine_Similarity_Cython(data_matrix, topK=TopK, normalize = True,
                                                     shrink=shrink, mode='jaccard')
        W_dense_Cython = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Python(data_matrix, topK=TopK, normalize = True,
                                                     shrink=shrink, mode='jaccard')
        W_dense_Python = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Parallel(data_matrix, topK=TopK, normalize = True,
                                                     shrink=shrink, mode='jaccard')
        W_dense_Parallel = cosine_similarity.compute_similarity()


        data_matrix.data = np.ones_like(data_matrix.data)
        data_matrix = data_matrix.toarray().astype(np.float64)

        W_dense_mul = data_matrix.T.dot(data_matrix)


        W_dense_denominator = np.matlib.repmat((data_matrix**2).sum(axis=0), data_matrix.shape[1], 1)
        W_dense_denominator = W_dense_denominator + W_dense_denominator.T - W_dense_mul + shrink

        W_dense_mul[W_dense_denominator>0] /= W_dense_denominator[W_dense_denominator>0]

        W_dense_mul[np.arange(W_dense_mul.shape[0]),np.arange(W_dense_mul.shape[0])] = 0.0

        assert np.allclose(W_dense_Cython, W_dense_mul, atol=1e-4), "W_dense_Cython not matching control"
        assert np.allclose(W_dense_Python, W_dense_mul, atol=1e-4), "W_dense_Python not matching control"
        assert np.allclose(W_dense_Parallel, W_dense_mul, atol=1e-4), "W_dense_Parallel not matching control"



    def test_cosine_similarity_dense_big(self):

        from Base.Cython.cosine_similarity import Cosine_Similarity as Cosine_Similarity_Cython
        from Base.cosine_similarity import Compute_Similarity as Cosine_Similarity_Python
        from Base.cosine_similarity_parallel import Cosine_Similarity_Parallel as Cosine_Similarity_Parallel

        TopK = 0
        n_items = 500
        n_users = 1000

        data_matrix = sps.random(n_users, n_items, density=0.1)

        cosine_similarity = Cosine_Similarity_Cython(data_matrix, topK=TopK, normalize = False)
        W_dense_Cython = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Python(data_matrix, topK=TopK, normalize = False)
        W_dense_Python = cosine_similarity.compute_similarity()

        cosine_similarity = Cosine_Similarity_Parallel(data_matrix, topK=TopK, normalize = False)
        W_dense_Parallel = cosine_similarity.compute_similarity()


        W_dense_mul = data_matrix.T.dot(data_matrix).toarray()
        W_dense_mul[np.arange(W_dense_mul.shape[0]),np.arange(W_dense_mul.shape[0])] = 0.0

        assert np.allclose(W_dense_Cython, W_dense_mul, atol=1e-4), "W_dense_Cython not matching control"
        assert np.allclose(W_dense_Python, W_dense_mul, atol=1e-4), "W_dense_Python not matching control"
        assert np.allclose(W_dense_Parallel, W_dense_mul, atol=1e-4), "W_dense_Parallel not matching control"


    def test_cosine_similarity_TopK(self):

        from Base.Cython.cosine_similarity import Cosine_Similarity as Cosine_Similarity_Cython
        from Base.cosine_similarity import Compute_Similarity as Cosine_Similarity_Python
        from Base.cosine_similarity_parallel import Cosine_Similarity_Parallel as Cosine_Similarity_Parallel

        TopK=4

        data_matrix = np.array([[1,1,0,1],[0,1,1,1],[1,0,1,0]])
        data_matrix = sps.csr_matrix(data_matrix)

        cosine_similarity = Cosine_Similarity_Cython(data_matrix, topK=TopK, normalize = False)
        W_dense_Cython = cosine_similarity.compute_similarity().toarray()

        cosine_similarity = Cosine_Similarity_Python(data_matrix, topK=TopK, normalize = False)
        W_dense_Python = cosine_similarity.compute_similarity().toarray()

        cosine_similarity = Cosine_Similarity_Parallel(data_matrix, topK=TopK, normalize = False)
        W_dense_Parallel = cosine_similarity.compute_similarity().toarray()


        W_dense_mul = data_matrix.T.dot(data_matrix)
        W_dense_mul[np.arange(W_dense_mul.shape[0]),np.arange(W_dense_mul.shape[0])] = 0.0

        W_dense_mul = similarityMatrixTopK(W_dense_mul, k=TopK).toarray()

        assert np.allclose(W_dense_Cython, W_dense_mul, atol=1e-4), "W_sparse_Cython not matching control"
        assert np.allclose(W_dense_Python, W_dense_mul, atol=1e-4), "W_dense_Python not matching control"
        assert np.allclose(W_dense_Parallel, W_dense_mul, atol=1e-4), "W_dense_Parallel not matching control"



    def test_cosine_similarity_TopK_big(self):

        from Base.Cython.cosine_similarity import Cosine_Similarity as Cosine_Similarity_Cython
        from Base.cosine_similarity import Compute_Similarity as Cosine_Similarity_Python
        from Base.cosine_similarity_parallel import Cosine_Similarity_Parallel as Cosine_Similarity_Parallel


        n_items = 500
        n_users = 1000
        TopK = n_items


        data_matrix = sps.random(n_users, n_items, density=0.1)

        cosine_similarity = Cosine_Similarity_Cython(data_matrix, topK=TopK, normalize = False)
        W_dense_Cython = cosine_similarity.compute_similarity().toarray()

        cosine_similarity = Cosine_Similarity_Python(data_matrix, topK=TopK, normalize = False)
        W_dense_Python = cosine_similarity.compute_similarity().toarray()

        cosine_similarity = Cosine_Similarity_Parallel(data_matrix, topK=TopK, normalize = False)
        W_dense_Parallel = cosine_similarity.compute_similarity().toarray()

        W_dense_mul = data_matrix.T.dot(data_matrix)
        W_dense_mul[np.arange(W_dense_mul.shape[0]),np.arange(W_dense_mul.shape[0])] = 0.0

        W_dense_mul = similarityMatrixTopK(W_dense_mul, k=TopK).toarray()

        assert np.allclose(W_dense_Cython, W_dense_mul, atol=1e-4), "W_sparse_Cython not matching control"
        assert np.allclose(W_dense_Python, W_dense_mul, atol=1e-4), "W_dense_Python not matching control"
        assert np.allclose(W_dense_Parallel, W_dense_mul, atol=1e-4), "W_dense_Parallel not matching control"




def runCompilationScript():

    # Run compile script setting the working directory to ensure the compiled file are contained in the
    # appropriate subfolder and not the project root

    compiledModuleSubfolder = "/Cython"
    fileToCompile = 'cosine_similarity.pyx'

    command = ['python',
               'compileCython.py',
               fileToCompile,
               'build_ext',
               '--inplace'
               ]

    output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)


    try:

        command = ['cython',
                   fileToCompile,
                   '-a'
                   ]

        output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

    except:
        pass

    print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

    # Command to run compilation script
    # python compileCython.py Compute_Similarity_Cython.pyx build_ext --inplace

    # Command to generate html report
    # cython -a Compute_Similarity_Cython.pyx

if __name__ == '__main__':

    runCompilationScript()

    unittest.main()