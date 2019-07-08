#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/09/2018

@author: Maurizio Ferrari Dacrema
"""


import scipy.sparse as sps

class IncrementalSparseMatrix_ListBased(object):

    def __init__(self, auto_create_col_mapper = False, auto_create_row_mapper = False, n_rows = None, n_cols = None):

        super(IncrementalSparseMatrix_ListBased, self).__init__()

        self._row_list = []
        self._col_list = []
        self._data_list = []

        self._n_rows = n_rows
        self._n_cols = n_cols
        self._auto_create_column_mapper = auto_create_col_mapper
        self._auto_create_row_mapper = auto_create_row_mapper

        if self._auto_create_column_mapper:
            self._column_original_ID_to_index = {}

        if self._auto_create_row_mapper:
            self._row_original_ID_to_index = {}


    def add_data_lists(self, row_list_to_add, col_list_to_add, data_list_to_add):

        assert len(row_list_to_add) == len(col_list_to_add) and len(row_list_to_add) == len(data_list_to_add),\
            "IncrementalSparseMatrix: element lists must have different length"


        col_list_index = [self._get_column_index(column_id) for column_id in col_list_to_add]
        row_list_index = [self._get_row_index(row_id) for row_id in row_list_to_add]

        self._row_list.extend(row_list_index)
        self._col_list.extend(col_list_index)
        self._data_list.extend(data_list_to_add)




    def add_single_row(self, row_id, col_list, data = 1.0):

        n_elements = len(col_list)

        col_list_index = [self._get_column_index(column_id) for column_id in col_list]
        row_index = self._get_row_index(row_id)

        self._row_list.extend([row_index] * n_elements)
        self._col_list.extend(col_list_index)
        self._data_list.extend([data] * n_elements)



    def get_column_token_to_id_mapper(self):

        if self._auto_create_column_mapper:
            return self._column_original_ID_to_index.copy()



        dummy_column_original_ID_to_index = {}

        for col in range(self._n_cols):
            dummy_column_original_ID_to_index[col] = col

        return dummy_column_original_ID_to_index



    def get_row_token_to_id_mapper(self):

        if self._auto_create_row_mapper:
            return self._row_original_ID_to_index.copy()



        dummy_row_original_ID_to_index = {}

        for row in range(self._n_rows):
            dummy_row_original_ID_to_index[row] = row

        return dummy_row_original_ID_to_index



    def _get_column_index(self, column_id):

        if not self._auto_create_column_mapper:
            column_index = column_id

        else:

            if column_id in self._column_original_ID_to_index:
                column_index = self._column_original_ID_to_index[column_id]

            else:
                column_index = len(self._column_original_ID_to_index)
                self._column_original_ID_to_index[column_id] = column_index

        return column_index


    def _get_row_index(self, row_id):

        if not self._auto_create_row_mapper:
            row_index = row_id

        else:

            if row_id in self._row_original_ID_to_index:
                row_index = self._row_original_ID_to_index[row_id]

            else:
                row_index = len(self._row_original_ID_to_index)
                self._row_original_ID_to_index[row_id] = row_index

        return row_index


    def get_nnz(self):
        return len(self._row_list)



    def get_SparseMatrix(self):

        if self._n_rows is None:
            self._n_rows = max(self._row_list) + 1

        if self._n_cols is None:
            self._n_cols = max(self._col_list) + 1

        shape = (self._n_rows, self._n_cols)

        sparseMatrix = sps.csr_matrix((self._data_list, (self._row_list, self._col_list)), shape=shape)
        sparseMatrix.eliminate_zeros()


        return sparseMatrix





import numpy as np



class IncrementalSparseMatrix(IncrementalSparseMatrix_ListBased):

    def __init__(self, auto_create_col_mapper = False, auto_create_row_mapper = False, n_rows = None, n_cols = None, dtype = np.float64):

        super(IncrementalSparseMatrix, self).__init__(auto_create_col_mapper = auto_create_col_mapper,
                                                             auto_create_row_mapper = auto_create_row_mapper,
                                                             n_rows = n_rows,
                                                             n_cols = n_cols)

        self._dataBlock = 10000000
        self._next_cell_pointer = 0

        self._dtype_data = dtype
        self._dtype_coordinates = np.uint32
        self._max_value_of_coordinate_dtype = np.iinfo(self._dtype_coordinates).max

        self._row_array = np.zeros(self._dataBlock, dtype=self._dtype_coordinates)
        self._col_array = np.zeros(self._dataBlock, dtype=self._dtype_coordinates)
        self._data_array = np.zeros(self._dataBlock, dtype=self._dtype_data)


    def get_nnz(self):
        return self._next_cell_pointer


    def add_data_lists(self, row_list_to_add, col_list_to_add, data_list_to_add):

        assert len(row_list_to_add) == len(col_list_to_add) and len(row_list_to_add) == len(data_list_to_add),\
            "IncrementalSparseMatrix: element lists must have the same length"

        for data_point_index in range(len(row_list_to_add)):

            if self._next_cell_pointer == len(self._row_array):
                self._row_array = np.concatenate((self._row_array, np.zeros(self._dataBlock, dtype=self._dtype_coordinates)))
                self._col_array = np.concatenate((self._col_array, np.zeros(self._dataBlock, dtype=self._dtype_coordinates)))
                self._data_array = np.concatenate((self._data_array, np.zeros(self._dataBlock, dtype=self._dtype_data)))


            row_index = self._get_row_index(row_list_to_add[data_point_index])
            col_index = self._get_column_index(col_list_to_add[data_point_index])

            self._row_array[self._next_cell_pointer] = row_index
            self._col_array[self._next_cell_pointer] = col_index
            self._data_array[self._next_cell_pointer] = data_list_to_add[data_point_index]

            self._next_cell_pointer += 1




    def add_single_row(self, row_index, col_list, data = 1.0):

        n_elements = len(col_list)

        self.add_data_lists([row_index] * n_elements,
                            col_list,
                            [data] * n_elements)





    def get_SparseMatrix(self):

        if self._n_rows is None:
            self._n_rows = self._row_array.max() + 1

        if self._n_cols is None:
            self._n_cols = self._col_array.max() + 1

        shape = (self._n_rows, self._n_cols)

        sparseMatrix = sps.csr_matrix((self._data_array[:self._next_cell_pointer],
                                       (self._row_array[:self._next_cell_pointer], self._col_array[:self._next_cell_pointer])),
                                      shape=shape,
                                      dtype=self._dtype_data)

        sparseMatrix.eliminate_zeros()


        return sparseMatrix








class IncrementalSparseMatrix_FilterIDs(IncrementalSparseMatrix):
    """
    This class builds an IncrementalSparseMatrix allowing to constrain the row and column IDs that will be added
    It is useful, for example, when
    """

    def __init__(self, preinitialized_col_mapper = None, preinitialized_row_mapper = None,
                 on_new_col = "add", on_new_row = "add", dtype = np.float64):
        """
        Possible behaviour is:
        - Automatically add new ids:    if_new_col = "add" and predefined_col_mapper = None or predefined_col_mapper = {dict}
        - Ignore new ids                if_new_col = "ignore" and predefined_col_mapper = {dict}
        :param preinitialized_col_mapper:
        :param preinitialized_row_mapper:
        :param on_new_col:
        :param on_new_row:
        :param n_rows:
        :param n_cols:
        """

        super(IncrementalSparseMatrix_FilterIDs, self).__init__(dtype = dtype)

        self._row_list = []
        self._col_list = []
        self._data_list = []

        assert on_new_col in ["add", "ignore"], "IncrementalSparseMatrix: if_new_col value not recognized, allowed values are 'add', 'ignore', provided was '{}'".format(on_new_col)
        assert on_new_row in ["add", "ignore"], "IncrementalSparseMatrix: if_new_row value not recognized, allowed values are 'add', 'ignore', provided was '{}'".format(on_new_row)

        if on_new_col == "add":
            assert preinitialized_col_mapper is None or isinstance(preinitialized_col_mapper, dict), "IncrementalSparseMatrix: if on_new_col is 'add' then preinitialized_col_mapper must be either 'None' or contain a dictionary"

        if on_new_row == "add":
            assert preinitialized_row_mapper is None or isinstance(preinitialized_row_mapper, dict), "IncrementalSparseMatrix: if on_new_row is 'add' then preinitialized_row_mapper must be either 'None' or contain a dictionary"

        if on_new_col == "ignore":
            assert isinstance(preinitialized_col_mapper, dict), "IncrementalSparseMatrix: if on_new_col is 'ignore' then preinitialized_col_mapper must be a dictionary"

        if on_new_row == "ignore":
            assert isinstance(preinitialized_row_mapper, dict), "IncrementalSparseMatrix: if on_new_row is 'ignore' then preinitialized_row_mapper must be a dictionary"


        self._on_new_col_add_flag = on_new_col == "add"
        self._on_new_row_add_flag = on_new_row == "add"

        self._auto_create_row_mapper = True
        self._auto_create_column_mapper = True


        if preinitialized_col_mapper is None:
            self._column_original_ID_to_index = {}
        else:
            self._column_original_ID_to_index = preinitialized_col_mapper.copy()

        if preinitialized_row_mapper is None:
            self._row_original_ID_to_index = {}
        else:
            self._row_original_ID_to_index = preinitialized_row_mapper.copy()




    def _get_column_index(self, column_id):

        if column_id in self._column_original_ID_to_index:
            column_index = self._column_original_ID_to_index[column_id]

        elif self._on_new_col_add_flag:
            column_index = len(self._column_original_ID_to_index)
            self._column_original_ID_to_index[column_id] = column_index

        else:
            column_index = None

        return column_index




    def _get_row_index(self, row_id):

        if row_id in self._row_original_ID_to_index:
            row_index = self._row_original_ID_to_index[row_id]

        elif self._on_new_row_add_flag:
            row_index = len(self._row_original_ID_to_index)
            self._row_original_ID_to_index[row_id] = row_index

        else:
            row_index = None

        return row_index




    def add_data_lists(self, row_list_to_add, col_list_to_add, data_list_to_add):

        assert len(row_list_to_add) == len(col_list_to_add) and len(row_list_to_add) == len(data_list_to_add),\
            "IncrementalSparseMatrix: element lists must have different length"


        for data_point_index in range(len(row_list_to_add)):

            if self._next_cell_pointer == len(self._row_array):
                self._row_array = np.concatenate((self._row_array, np.zeros(self._dataBlock, dtype=self._dtype_coordinates)))
                self._col_array = np.concatenate((self._col_array, np.zeros(self._dataBlock, dtype=self._dtype_coordinates)))
                self._data_array = np.concatenate((self._data_array, np.zeros(self._dataBlock, dtype=self._dtype_data)))


            row_index = self._get_row_index(row_list_to_add[data_point_index])
            col_index = self._get_column_index(col_list_to_add[data_point_index])


            if row_index is not None and col_index is not None:

                self._row_array[self._next_cell_pointer] = row_index
                self._col_array[self._next_cell_pointer] = col_index
                self._data_array[self._next_cell_pointer] = data_list_to_add[data_point_index]

                self._next_cell_pointer += 1



    def get_SparseMatrix(self):

        # Set fixed dimension len to ensure that the matrix is not smaller than the number of entries in the dictionary
        self._n_rows = len(self._row_original_ID_to_index)
        self._n_cols = len(self._column_original_ID_to_index)


        return super(IncrementalSparseMatrix_FilterIDs, self).get_SparseMatrix()