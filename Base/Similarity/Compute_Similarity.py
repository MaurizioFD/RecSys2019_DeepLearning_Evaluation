#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/06/18

@author: Anonymous authors
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

        self.dense = False

        if similarity == "euclidean":
            # This is only available here
            self.compute_similarity_object = Compute_Similarity_Euclidean(dataMatrix, **args)

        else:

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


#
#
#
#
# class Compute_Similarity:
#
#     COSINE = "cosine"
#     JACCARD = "jaccard"
#     AS_COSINE = "as_cosine"
#     P_3_AlPHA = "p3aplha"
#     TVERSKY = "tversky"
#     R_P_3_BETA = "rp3beta"
#     TANIMOTO = "tanimoto"
#     DICE = "dice"
#
#
#
#     def __init__(self, dataMatrix, use_implementation = "density", is_binary = False, **args):
#
#         self.dataMatrix = dataMatrix.copy()
#         self.use_implementation = use_implementation
#
#
#
#         if "mode" in args:
#             similarity_mode = args["mode"]
#         else:
#             similarity_mode = self.COSINE
#
#         self.similarity_mode = similarity_mode
#
#
#         if self.use_implementation == "density":
#
#             if isinstance(dataMatrix, np.ndarray):
#                 self.dense = True
#
#             elif isinstance(dataMatrix, sps.spmatrix):
#                 shape = dataMatrix.shape
#
#                 num_cells = shape[0]*shape[1]
#
#                 sparsity = dataMatrix.nnz/num_cells
#
#                 self.dense = sparsity > 0.5
#
#             else:
#
#                 print("Cosine_Similarity: matrix type not recognized, calling default...")
#
#             if self.dense:
#
#                 print("Cosine_Similarity: detected dense matrix")
#                 self.use_implementation = "python"
#
#             else:
#                 self.use_implementation = "cython"
#
#
#
#
#         if self.use_implementation == "cython":
#
#             try:
#
#                 self.cython_args = {"k": args["topK"],
#                                "shrink": args["shrink"],
#                                "threshold":0,
#                                "binary": is_binary}
#
#
#
#                 from Base.Similarity.Cython.cosine import cosine_similarity
#                 from Base.Similarity.Cython.dice import dice_similarity
#                 from Base.Similarity.Cython.tversky import tversky_similarity
#                 from Base.Similarity.Cython.jaccard import tanimoto_similarity, jaccard_similarity
#
#
#
#             except ImportError:
#                 print("Unable to load Cython Cosine_Similarity, reverting to Python")
#                 self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)
#
#
#         elif use_implementation == "python":
#             self.compute_similarity_object = Compute_Similarity_Python(dataMatrix, **args)
#
#         else:
#
#             raise  ValueError("Cosine_Similarity: value for argument 'use_implementation' not recognized")
#
#
#
#
#
#     def compute_similarity(self,  **args):
#
#
#         if self.use_implementation == "cython":
#
#             from Base.Similarity.Cython.cosine import cosine_similarity
#             from Base.Similarity.Cython.dice import dice_similarity
#             from Base.Similarity.Cython.tversky import tversky_similarity
#             from Base.Similarity.Cython.jaccard import tanimoto_similarity, jaccard_similarity
#
#             if self.similarity_mode == self.COSINE:
#                 W_sparse = cosine_similarity(self.dataMatrix.T, **self.cython_args)
#
#             elif self.similarity_mode == self.JACCARD:
#                 W_sparse = jaccard_similarity(self.dataMatrix.T, **self.cython_args)
#
#             elif self.similarity_mode == self.TANIMOTO:
#                 W_sparse = tanimoto_similarity(self.dataMatrix.T, **self.cython_args)
#
#             elif self.similarity_mode == self.AS_COSINE:
#                 W_sparse = cosine_similarity(self.dataMatrix.T, alpha=alpha, k=top_k, shrink=shrink)
#
#             elif self.similarity_mode == self.DICE:
#                 W_sparse = dice_similarity(self.dataMatrix.T, **self.cython_args)
#
#             elif self.similarity_mode == self.TVERSKY:
#                 W_sparse = tversky_similarity(self.dataMatrix.T, k=top_k, alpha=alpha, beta=beta, shrink=shrink, binary=binary)
#
#             return sps.csr_matrix(W_sparse)
#
#
#         return self.compute_similarity_object.compute_similarity(**args)
#
