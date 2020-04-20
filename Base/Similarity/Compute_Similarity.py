#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/06/18

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps

from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from Base.Similarity.Compute_Similarity_Euclidean import Compute_Similarity_Euclidean


from enum import Enum

class SimilarityFunction(Enum):
    COSINE = "cosine"
    PEARSON = "pearson"
    JACCARD = "jaccard"
    TANIMOTO = "tanimoto"
    ADJUSTED_COSINE = "adjusted"
    EUCLIDEAN = "euclidean"




class Compute_Similarity:


    def __init__(self, dataMatrix, use_implementation = "density", similarity = None, **args):
        """
        Interface object that will call the appropriate similarity implementation
        :param dataMatrix:
        :param use_implementation:      "density" will choose the most efficient implementation automatically
                                        "cython" will use the cython implementation, if available. Most efficient for sparse matrix
                                        "python" will use the python implementation. Most efficent for dense matrix
        :param similarity:              the type of similarity to use, see SimilarityFunction enum
        :param args:                    other args required by the specific similarity implementation
        """

        assert np.all(np.isfinite(dataMatrix.data)), \
            "Compute_Similarity: Data matrix contains {} non finite values".format(np.sum(np.logical_not(np.isfinite(dataMatrix.data))))

        self.dense = False

        if similarity == "euclidean":
            # This is only available here
            self.compute_similarity_object = Compute_Similarity_Euclidean(dataMatrix, **args)

        else:

            assert not (dataMatrix.shape[0] == 1 and dataMatrix.nnz == dataMatrix.shape[1]),\
                "Compute_Similarity: data has only 1 feature (shape: {}) with dense values," \
                " vector and set based similarities are not defined on 1-dimensional dense data," \
                " use Euclidean similarity instead.".format(dataMatrix.shape)

            if similarity is not None:
                args["similarity"] = similarity


            if use_implementation == "density":

                if isinstance(dataMatrix, np.ndarray):
                    self.dense = True

                elif isinstance(dataMatrix, sps.spmatrix):
                    shape = dataMatrix.shape

                    num_cells = shape[0]*shape[1]

                    sparsity = dataMatrix.nnz/num_cells

                    self.dense = sparsity > 0.5

                else:
                    print("Compute_Similarity: matrix type not recognized, calling default...")
                    use_implementation = "python"

                if self.dense:
                    print("Compute_Similarity: detected dense matrix")
                    use_implementation = "python"
                else:
                    use_implementation = "cython"





            if use_implementation == "cython":

                try:
                    from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
                    self.compute_similarity_object = Compute_Similarity_Cython(dataMatrix, **args)

                except ImportError:
                    print("Unable to load Cython Compute_Similarity, reverting to Python")
                    self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)


            elif use_implementation == "python":
                self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)

            else:

                raise  ValueError("Compute_Similarity: value for argument 'use_implementation' not recognized")





    def compute_similarity(self,  **args):

        return self.compute_similarity_object.compute_similarity(**args)

