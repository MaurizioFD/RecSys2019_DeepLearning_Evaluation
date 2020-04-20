#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
import time
import os

def check_matrix(X, format='csc', dtype=np.float32):
    """
    This function takes a matrix as input and transforms it into the specified format.
    The matrix in input can be either sparse or ndarray.
    If the matrix in input has already the desired format, it is returned as-is
    the dtype parameter is always applied and the default is np.float32
    :param X:
    :param format:
    :param dtype:
    :return:
    """


    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)

    elif format == 'npy':
        if sps.issparse(X):
            return X.toarray().astype(dtype)
        else:
            return np.array(X)

    elif isinstance(X, np.ndarray):
        X = sps.csr_matrix(X, dtype=dtype)
        X.eliminate_zeros()
        return check_matrix(X, format=format, dtype=dtype)
    else:
        return X.astype(dtype)


def similarityMatrixTopK(item_weights, k=100, verbose = False):
    """
    The function selects the TopK most similar elements, column-wise

    :param item_weights:
    :param forceSparseOutput:
    :param k:
    :param verbose:
    :param inplace: Default True, WARNING matrix will be modified
    :return:
    """

    assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

    start_time = time.time()

    if verbose:
        print("Generating topK matrix")

    nitems = item_weights.shape[1]
    k = min(k, nitems)

    # for each column, keep only the top-k scored items
    sparse_weights = not isinstance(item_weights, np.ndarray)

    # iterate over each column and keep only the top-k similar items
    data, rows_indices, cols_indptr = [], [], []

    if sparse_weights:
        item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)
    else:
        column_row_index = np.arange(nitems, dtype=np.int32)



    for item_idx in range(nitems):

        cols_indptr.append(len(data))

        if sparse_weights:
            start_position = item_weights.indptr[item_idx]
            end_position = item_weights.indptr[item_idx+1]

            column_data = item_weights.data[start_position:end_position]
            column_row_index = item_weights.indices[start_position:end_position]

        else:
            column_data = item_weights[:,item_idx]


        non_zero_data = column_data!=0

        idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
        top_k_idx = idx_sorted[-k:]

        data.extend(column_data[non_zero_data][top_k_idx])
        rows_indices.extend(column_row_index[non_zero_data][top_k_idx])


    cols_indptr.append(len(data))

    # During testing CSR is faster
    W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)

    if verbose:
        print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

    return W_sparse




def areURMequals(URM1, URM2):

    if(URM1.shape != URM2.shape):
        return False

    return (URM1-URM2).nnz ==0


def removeTopPop(URM_1, URM_2=None, percentageToRemove=0.2):
    """
    Remove the top popular items from the matrix
    :param URM_1: user X items
    :param URM_2: user X items
    :param percentageToRemove: value 1 corresponds to 100%
    :return: URM: user X selectedItems, obtained from URM_1
             Array: itemMappings[selectedItemIndex] = originalItemIndex
             Array: removedItems
    """


    item_pop = URM_1.sum(axis=0)  # this command returns a numpy.matrix of size (1, nitems)

    if URM_2 != None:

        assert URM_2.shape[1] == URM_1.shape[1], \
            "The two URM do not contain the same number of columns, URM_1 has {}, URM_2 has {}".format(URM_1.shape[1], URM_2.shape[1])

        item_pop += URM_2.sum(axis=0)


    item_pop = np.asarray(item_pop).squeeze()  # necessary to convert it into a numpy.array of size (nitems,)
    popularItemsSorted = np.argsort(item_pop)[::-1]

    numItemsToRemove = int(len(popularItemsSorted)*percentageToRemove)

    # Choose which columns to keep
    itemMask = np.in1d(np.arange(len(popularItemsSorted)), popularItemsSorted[:numItemsToRemove],  invert=True)

    # Map the column index of the new URM to the original ItemID
    itemMappings = np.arange(len(popularItemsSorted))[itemMask]

    removedItems = np.arange(len(popularItemsSorted))[np.logical_not(itemMask)]

    return URM_1[:,itemMask], itemMappings, removedItems


def addZeroSamples(S_matrix, numSamplesToAdd):

    n_items = S_matrix.shape[1]

    S_matrix_coo = S_matrix.tocoo()

    row_index = list(S_matrix_coo.row)
    col_index = list(S_matrix_coo.col)
    data = list(S_matrix_coo.data)

    existingSamples = set(zip(row_index, col_index))

    addedSamples = 0
    consecutiveFailures = 0

    while (addedSamples < numSamplesToAdd):

        item1 = np.random.randint(0, n_items)
        item2 = np.random.randint(0, n_items)

        if (item1 != item2 and (item1, item2) not in existingSamples):

            row_index.append(item1)
            col_index.append(item2)
            data.append(0)

            existingSamples.add((item1, item2))

            addedSamples += 1
            consecutiveFailures = 0

        else:
            consecutiveFailures += 1

        if (consecutiveFailures >= 100):
            raise SystemExit(
                "Unable to generate required zero samples, termination at 100 consecutive discarded samples")

    return row_index, col_index, data


def reshapeSparse(sparseMatrix, newShape):

    if sparseMatrix.shape[0] > newShape[0] or sparseMatrix.shape[1] > newShape[1]:
        ValueError("New shape cannot be smaller than SparseMatrix. SparseMatrix shape is: {}, newShape is {}".format(
            sparseMatrix.shape, newShape))


    sparseMatrix = sparseMatrix.tocoo()
    newMatrix = sps.csr_matrix((sparseMatrix.data, (sparseMatrix.row, sparseMatrix.col)), shape=newShape)

    return newMatrix









def get_unique_temp_folder(input_temp_folder_path):
    """
    The function returns the path of a folder in result_experiments
    The function guarantees that the folder is not already existent and it creates it
    :return:
    """

    if input_temp_folder_path[-1] == "/":
        input_temp_folder_path = input_temp_folder_path[:-1]

    progressive_temp_folder_name = input_temp_folder_path

    counter_suffix = 0

    while os.path.isdir(progressive_temp_folder_name):

        counter_suffix += 1
        progressive_temp_folder_name = input_temp_folder_path + "_" + str(counter_suffix)


    progressive_temp_folder_name += "/"
    os.makedirs(progressive_temp_folder_name)

    return progressive_temp_folder_name
