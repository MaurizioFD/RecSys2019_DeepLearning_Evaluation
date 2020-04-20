"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


"""
Determine the operative system. The interface of numpy returns a different type for argsort under windows and linux

http://docs.cython.org/en/latest/src/userguide/language_basics.html#conditional-compilation
"""
IF UNAME_SYSNAME == "linux":
    DEF LONG_t = "long"
ELIF  UNAME_SYSNAME == "Windows":
    DEF LONG_t = "long long"
ELSE:
    DEF LONG_t = "long long"




from Base.Recommender_utils import similarityMatrixTopK, check_matrix
import numpy as np
import cython
cimport numpy as np
import time
import sys

from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, srand, RAND_MAX


cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item
    long seen_items_start_pos
    long seen_items_end_pos


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef class SLIM_BPR_Cython_Epoch:

    cdef int n_users, n_items, batch_size
    cdef int topK
    cdef int symmetric, train_with_sparse_weights, final_model_sparse_weights

    cdef double learning_rate, li_reg, lj_reg

    cdef int[:] URM_mask_indices, URM_mask_indptr

    cdef Sparse_Matrix_Tree_CSR S_sparse
    cdef Triangular_Matrix S_symmetric
    cdef double[:,:] S_dense


    # Adaptive gradient

    cdef int useAdaGrad, useRmsprop, useAdam, verbose

    cdef double [:] sgd_cache_I
    cdef double gamma

    cdef double [:] sgd_cache_I_momentum_1, sgd_cache_I_momentum_2
    cdef double beta_1, beta_2, beta_1_power_t, beta_2_power_t
    cdef double momentum_1, momentum_2



    def __init__(self, URM_mask,
                 train_with_sparse_weights = False,
                 final_model_sparse_weights = True,
                 learning_rate = 0.01, li_reg = 0.0, lj_reg = 0.0,
                 batch_size = 1, topK = 150, symmetric = True,
                 verbose = False, random_seed = None,
                 sgd_mode='adam', gamma=0.995, beta_1=0.9, beta_2=0.999):

        super(SLIM_BPR_Cython_Epoch, self).__init__()

        # Create copy of URM_train in csr format
        # make sure indices are sorted
        URM_mask = check_matrix(URM_mask, 'csr')
        URM_mask = URM_mask.sorted_indices()

        self.n_users = URM_mask.shape[0]
        self.n_items = URM_mask.shape[1]
        self.topK = topK
        self.learning_rate = learning_rate
        self.li_reg = li_reg
        self.lj_reg = lj_reg
        self.batch_size = batch_size
        self.verbose = verbose


        if train_with_sparse_weights:
            symmetric = False

        self.train_with_sparse_weights = train_with_sparse_weights
        self.final_model_sparse_weights = final_model_sparse_weights
        self.symmetric = symmetric

        self.URM_mask_indices = np.array(URM_mask.indices, dtype=np.int32)
        self.URM_mask_indptr = np.array(URM_mask.indptr, dtype=np.int32)


        if self.train_with_sparse_weights:
            self.S_sparse = Sparse_Matrix_Tree_CSR(self.n_items, self.n_items)

        elif self.symmetric:
            self.S_symmetric = Triangular_Matrix(self.n_items, isSymmetric = True)
        else:
            self.S_dense = np.zeros((self.n_items, self.n_items), dtype=np.float64)

        if random_seed is not None:
            srand(<unsigned> random_seed)


        self._init_adaptive_gradient_cache(sgd_mode, gamma, beta_1, beta_2)




    def _init_adaptive_gradient_cache(self, sgd_mode, gamma, beta_1, beta_2):

        self.useAdaGrad = False
        self.useRmsprop = False
        self.useAdam = False

        if sgd_mode=='adagrad':
            self.useAdaGrad = True

        elif sgd_mode=='rmsprop':
            self.useRmsprop = True

            # Gamma default value suggested by Hinton
            # self.gamma = 0.9
            self.gamma = gamma

        elif sgd_mode=='adam':
            self.useAdam = True

            # Default value suggested by the original paper
            # beta_1=0.9, beta_2=0.999
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.beta_1_power_t = beta_1
            self.beta_2_power_t = beta_2



        if sgd_mode=='sgd':
            self.sgd_cache_I = None

            self.sgd_cache_I_momentum_1 = None
            self.sgd_cache_I_momentum_2 = None

        else:

            # Adagrad and RMSProp
            self.sgd_cache_I = np.zeros((self.n_items), dtype=np.float64)

            # Adam
            self.sgd_cache_I_momentum_1 = np.zeros((self.n_items), dtype=np.float64)
            self.sgd_cache_I_momentum_2 = np.zeros((self.n_items), dtype=np.float64)




    def __dealloc__(self):
        """
        Remove all PyMalloc allocaded memory
        :return:
        """

        self._dealloc()


    def _dealloc(self):

        print("Deallocating Cython objects")

        if self.S_sparse is not None:
            self.S_sparse.dealloc()
            self.S_sparse = None

        if self.S_symmetric is not None:
            self.S_symmetric.dealloc()
            self.S_symmetric = None






    def epochIteration_Cython(self):

        # Get number of available interactions
        cdef long totalNumberOfBatch = int(self.n_users / self.batch_size) + 1

        cdef long start_time_epoch = time.time()
        cdef long start_time_batch = time.time()

        cdef BPR_sample sample
        cdef long i, j
        cdef long index, seenItem, numCurrentBatch, itemId
        cdef double x_uij, gradient, loss = 0.0
        cdef double local_gradient_i, local_gradient_j

        cdef int numSeenItems
        cdef int printStep

        if self.train_with_sparse_weights:
            printStep = 500000
        else:
            printStep = 5000000


        # Uniform user sampling without replacement
        for numCurrentBatch in range(totalNumberOfBatch):

            sample = self.sampleBPR_Cython()

            i = sample.pos_item
            j = sample.neg_item

            x_uij = 0.0

            # The difference is computed on the user_seen items

            index = 0
            while index <  sample.seen_items_end_pos - sample.seen_items_start_pos:

                seenItem = self.URM_mask_indices[sample.seen_items_start_pos + index]
                index +=1

                if self.train_with_sparse_weights:
                   x_uij += self.S_sparse.get_value(i, seenItem) - self.S_sparse.get_value(j, seenItem)

                elif self.symmetric:
                    x_uij += self.S_symmetric.get_value(i, seenItem) - self.S_symmetric.get_value(j, seenItem)

                else:
                    x_uij += self.S_dense[i, seenItem] - self.S_dense[j, seenItem]


            gradient = 1 / (1 + exp(x_uij))
            loss += x_uij**2


            local_gradient_i = self.adaptive_gradient(gradient, i, self.sgd_cache_I, self.sgd_cache_I_momentum_1, self.sgd_cache_I_momentum_2)
            local_gradient_j = self.adaptive_gradient(gradient, j, self.sgd_cache_I, self.sgd_cache_I_momentum_1, self.sgd_cache_I_momentum_2)


            index = 0
            while index < sample.seen_items_end_pos - sample.seen_items_start_pos:

                seenItem = self.URM_mask_indices[sample.seen_items_start_pos + index]
                index +=1

                if self.train_with_sparse_weights:
                    # Since the sparse matrix is slower compared to the others
                    # If no reg is required, avoid accessing it

                    if seenItem != i:
                        if self.li_reg!= 0.0:
                            self.S_sparse.add_value(i, seenItem, self.learning_rate * (local_gradient_i - self.li_reg * self.S_sparse.get_value(i, seenItem)))
                        else:
                            self.S_sparse.add_value(i, seenItem, self.learning_rate * local_gradient_i)


                    if seenItem != j:
                        if self.lj_reg!= 0.0:
                            self.S_sparse.add_value(j, seenItem, -self.learning_rate * (local_gradient_j - self.lj_reg * self.S_sparse.get_value(j, seenItem)))
                        else:
                            self.S_sparse.add_value(j, seenItem, -self.learning_rate * local_gradient_j)


                elif self.symmetric:

                    if seenItem != i:
                        self.S_symmetric.add_value(i, seenItem, self.learning_rate * (local_gradient_i - self.li_reg * self.S_symmetric.get_value(i, seenItem)))

                    if seenItem != j:
                        self.S_symmetric.add_value(j, seenItem, -self.learning_rate * (local_gradient_j - self.lj_reg * self.S_symmetric.get_value(j, seenItem)))

                else:

                    if seenItem != i:
                        self.S_dense[i, seenItem] += self.learning_rate * (local_gradient_i - self.li_reg * self.S_dense[i, seenItem])

                    if seenItem != j:
                        self.S_dense[j, seenItem] -= self.learning_rate * (local_gradient_j - self.lj_reg * self.S_dense[j, seenItem])



            # Exponentiation of beta at the end of each sample
            if self.useAdam:

                self.beta_1_power_t *= self.beta_1
                self.beta_2_power_t *= self.beta_2



            # If I have reached at least 20% of the total number of batches or samples
            # This allows to limit the memory occupancy of the sparse matrix
            if self.train_with_sparse_weights and numCurrentBatch % (totalNumberOfBatch/5) == 0 and numCurrentBatch!=0:
                self.S_sparse.rebalance_tree(TopK=self.topK)


            if self.verbose and ((numCurrentBatch%printStep==0 and not numCurrentBatch==0) or numCurrentBatch==totalNumberOfBatch-1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. BPR loss is {:.2E}. Sample per second: {:.0f}".format(
                    numCurrentBatch*self.batch_size,
                    100.0* float(numCurrentBatch*self.batch_size)/self.n_users,
                    time.time() - start_time_batch,
                    loss/(numCurrentBatch*self.batch_size + 1),
                    float(numCurrentBatch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()




    def get_S(self):

        # FIll diagonal with zeros
        cdef int index = 0

        while index < self.n_items:

            if self.train_with_sparse_weights:
                self.S_sparse.add_value(index, index, -self.S_sparse.get_value(index, index))

            elif self.symmetric:
                self.S_symmetric.add_value(index, index, -self.S_symmetric.get_value(index, index))

            else:
                self.S_dense[index, index] = 0.0

            index+=1



        if self.topK == False:

            if self.train_with_sparse_weights:
                return self.S_sparse.get_scipy_csr(TopK = False)

            elif self.symmetric:
                return self.S_symmetric.get_scipy_csr(TopK = False)

            else:

                if self.final_model_sparse_weights:
                    return similarityMatrixTopK(np.array(self.S_dense.T), k=self.topK).T
                else:
                    return np.array(self.S_dense)


        else :

            if self.train_with_sparse_weights:
                return self.S_sparse.get_scipy_csr(TopK=self.topK)

            elif self.symmetric:
                return self.S_symmetric.get_scipy_csr(TopK=self.topK)

            else:
                if self.final_model_sparse_weights:
                    return similarityMatrixTopK(np.array(self.S_dense.T), k=self.topK).T
                else:
                    return np.array(self.S_dense)






    cdef double adaptive_gradient(self, double gradient, long user_or_item_id, double[:] sgd_cache, double[:] sgd_cache_momentum_1, double[:] sgd_cache_momentum_2):


        cdef double gradient_update

        if self.useAdaGrad:
            sgd_cache[user_or_item_id] += gradient ** 2

            gradient_update = gradient / (sqrt(sgd_cache[user_or_item_id]) + 1e-8)


        elif self.useRmsprop:
            sgd_cache[user_or_item_id] = sgd_cache[user_or_item_id] * self.gamma + (1 - self.gamma) * gradient ** 2

            gradient_update = gradient / (sqrt(sgd_cache[user_or_item_id]) + 1e-8)


        elif self.useAdam:

            sgd_cache_momentum_1[user_or_item_id] = \
                sgd_cache_momentum_1[user_or_item_id] * self.beta_1 + (1 - self.beta_1) * gradient

            sgd_cache_momentum_2[user_or_item_id] = \
                sgd_cache_momentum_2[user_or_item_id] * self.beta_2 + (1 - self.beta_2) * gradient**2


            self.momentum_1 = sgd_cache_momentum_1[user_or_item_id]/ (1 - self.beta_1_power_t)
            self.momentum_2 = sgd_cache_momentum_2[user_or_item_id]/ (1 - self.beta_2_power_t)

            gradient_update = self.momentum_1/ (sqrt(self.momentum_2) + 1e-8)


        else:

            gradient_update = gradient



        return gradient_update


    cdef BPR_sample sampleBPR_Cython(self):

        cdef BPR_sample sample = BPR_sample(-1,-1,-1,-1,-1)

        cdef long index

        cdef int neg_item_selected, n_seen_items = 0


        # Skip users with no interactions or with no negative items
        while n_seen_items == 0 or n_seen_items == self.n_items:

            sample.user = rand() % self.n_users

            sample.seen_items_start_pos = self.URM_mask_indptr[sample.user]
            sample.seen_items_end_pos = self.URM_mask_indptr[sample.user + 1]

            n_seen_items = sample.seen_items_end_pos - sample.seen_items_start_pos


        index = rand() % n_seen_items

        sample.pos_item = self.URM_mask_indices[sample.seen_items_start_pos + index]


        neg_item_selected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        # for every user
        while not neg_item_selected:

            sample.neg_item = rand() % self.n_items

            index = 0
            # Indices data is sorted, so I don't need to go to the end of the current row
            while index < n_seen_items and self.URM_mask_indices[sample.seen_items_start_pos + index] < sample.neg_item:
                index+=1

            # If the positive item in position 'index' is == sample.neg_item, negative not selected
            # If the positive item in position 'index' is > sample.neg_item or index == n_seen_items, negative selected
            if index == n_seen_items or self.URM_mask_indices[sample.seen_items_start_pos + index] > sample.neg_item:
                neg_item_selected = True


        return sample



##################################################################################################################
#####################
#####################            SPARSE MATRIX
#####################
##################################################################################################################

import scipy.sparse as sps

import numpy as np
cimport numpy as np

#from libc.stdlib cimport malloc, free#, qsort
# PyMem malloc and free are slightly faster than plain C equivalents as they optimize OS calls
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# Declaring QSORT as "gil safe", appending "nogil" at the end of the declaration
# Otherwise I will not be able to pass the comparator function pointer
# https://stackoverflow.com/questions/8353076/how-do-i-pass-a-pointer-to-a-c-function-in-cython
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil


# Node struct
ctypedef struct matrix_element_tree_s:
    long column
    double data
    matrix_element_tree_s *higher
    matrix_element_tree_s *lower

ctypedef struct head_pointer_tree_s:
    matrix_element_tree_s *head


# Function to allocate a new node
cdef matrix_element_tree_s * pointer_new_matrix_element_tree_s(long column, double data, matrix_element_tree_s *higher,  matrix_element_tree_s *lower):

    cdef matrix_element_tree_s * new_element

    new_element = < matrix_element_tree_s * > PyMem_Malloc(sizeof(matrix_element_tree_s))
    new_element.column = column
    new_element.data = data
    new_element.higher = higher
    new_element.lower = lower

    return new_element


# Functions to compare structs to be used in C qsort
cdef int compare_struct_on_column(const void *a_input, const void *b_input):
    """
    The function compares the column contained in the two struct passed.
    If a.column > b.column returns >0  
    If a.column < b.column returns <0      
    
    :return int: a.column - b.column
    """

    cdef head_pointer_tree_s *a_casted = <head_pointer_tree_s *> a_input
    cdef head_pointer_tree_s *b_casted = <head_pointer_tree_s *> b_input

    return a_casted.head.column  - b_casted.head.column



cdef int compare_struct_on_data(const void * a_input, const void * b_input):
    """
    The function compares the data contained in the two struct passed.
    If a.data > b.data returns >0  
    If a.data < b.data returns <0      
    
    :return int: +1 or -1
    """

    cdef head_pointer_tree_s * a_casted = <head_pointer_tree_s *> a_input
    cdef head_pointer_tree_s * b_casted = <head_pointer_tree_s *> b_input

    if (a_casted.head.data - b_casted.head.data) > 0.0:
        return +1
    else:
        return -1



#################################
#################################       CLASS DECLARATION
#################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef class Sparse_Matrix_Tree_CSR:

    cdef long num_rows, num_cols

    # Array containing the struct (object, not pointer) corresponding to the root of the tree
    cdef head_pointer_tree_s* row_pointer


    def __init__(self, long num_rows, long num_cols):

        self.num_rows = num_rows
        self.num_cols = num_cols

        self.row_pointer = < head_pointer_tree_s *> PyMem_Malloc(self.num_rows * sizeof(head_pointer_tree_s))

        # Initialize all rows to empty
        for index in range(self.num_rows):
            self.row_pointer[index].head = NULL



    def dealloc(self):
        """
        Remove all PyMalloc allocaded memory
        :return:
        """

        cdef int numRow

        # Free all rows memory
        for numRow in range(self.num_rows):
            self.subtree_free_memory(self.row_pointer[numRow].head)

        PyMem_Free(self.row_pointer)






    cdef double add_value(self, long row, long col, double value):
        """
        The function adds a value to the specified cell. A new cell is created if necessary.         
        
        :param row: cell coordinates
        :param col:  cell coordinates
        :param value: value to add
        :return double: resulting cell value
        """

        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError("Cell is outside matrix. Matrix shape is ({},{}), coordinates given are ({},{})".format(
                self.num_rows, self.num_cols, row, col))

        cdef matrix_element_tree_s* current_element, new_element, * old_element
        cdef int stopSearch = False


        # If the row is empty, create a new element
        if self.row_pointer[row].head == NULL:

            # row_pointer is a python object, so I need the object itself and not the address
            self.row_pointer[row].head = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value


        # If the row is not empty, look for the cell
        # row_pointer contains the struct itself, but I just want its address
        current_element = self.row_pointer[row].head

        # Follow the tree structure
        while not stopSearch:

            if current_element.column < col and current_element.higher != NULL:
                current_element = current_element.higher

            elif current_element.column > col and current_element.lower != NULL:
                current_element = current_element.lower

            else:
                stopSearch = True

        # If the cell exist, update its value
        if current_element.column == col:
            current_element.data += value

            return current_element.data


        # The cell is not found, create new Higher element
        elif current_element.column < col and current_element.higher == NULL:
            current_element.higher = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)
            return value

        # The cell is not found, create new Lower element
        elif current_element.column > col and current_element.lower == NULL:
            current_element.lower = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)
            return value

        else:
            assert False, 'ERROR - Current insert operation is not implemented'




    cdef double get_value(self, long row, long col):
        """
        The function returns the value of the specified cell.         
        
        :param row: cell coordinates
        :param col:  cell coordinates
        :return double: cell value
        """


        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError(
                "Cell is outside matrix. Matrix shape is ({},{}), coordinates given are ({},{})".format(
                    self.num_rows, self.num_cols, row, col))


        cdef matrix_element_tree_s* current_element
        cdef int stopSearch = False

        # If the row is empty, return default
        if self.row_pointer[row].head == NULL:
            return 0.0


        # If the row is not empty, look for the cell
        # row_pointer contains the struct itself, but I just want its address
        current_element = self.row_pointer[row].head

        # Follow the tree structure
        while not stopSearch:

            if current_element.column < col and current_element.higher != NULL:
                current_element = current_element.higher

            elif current_element.column > col and current_element.lower != NULL:
                current_element = current_element.lower

            else:
                stopSearch = True


        # If the cell exist, return its value
        if current_element.column == col:
            return current_element.data

        # The cell is not found, return default
        else:
            return 0.0




    cdef get_scipy_csr(self, long TopK = False):
        """
        The function returns the current sparse matrix as a scipy_csr object         
   
        :return double: scipy_csr object
        """
        cdef int terminate
        cdef long row

        data = []
        indices = []
        indptr = []

        # Loop the rows
        for row in range(self.num_rows):

            #Always set indptr
            indptr.append(len(data))

            # row contains data
            if self.row_pointer[row].head != NULL:

                # Flatten the data structure
                self.row_pointer[row].head = self.subtree_to_list_flat(self.row_pointer[row].head)

                if TopK:
                    self.row_pointer[row].head = self.topK_selection_from_list(self.row_pointer[row].head, TopK)


                # Flatten the tree data
                subtree_column, subtree_data = self.from_linked_list_to_python_list(self.row_pointer[row].head)
                data.extend(subtree_data)
                indices.extend(subtree_column)

                # Rebuild the tree
                self.row_pointer[row].head = self.build_tree_from_list_flat(self.row_pointer[row].head)


        #Set terminal indptr
        indptr.append(len(data))

        return sps.csr_matrix((data, indices, indptr), shape=(self.num_rows, self.num_cols))



    cdef rebalance_tree(self, long TopK = False):
        """
        The function builds a balanced binary tree from the current one, for all matrix rows
        
        :param TopK: either False or an integer number. Number of the highest elements to preserve
        """

        cdef long row

        for row in range(self.num_rows):

            if self.row_pointer[row].head != NULL:

                # Flatten the data structure
                self.row_pointer[row].head = self.subtree_to_list_flat(self.row_pointer[row].head)

                if TopK:
                    self.row_pointer[row].head = self.topK_selection_from_list(self.row_pointer[row].head, TopK)

                # Rebuild the tree
                self.row_pointer[row].head = self.build_tree_from_list_flat(self.row_pointer[row].head)


    cdef matrix_element_tree_s * subtree_to_list_flat(self, matrix_element_tree_s * root):
        """
        The function flatten the structure of the subtree whose root is passed as a parameter    
        The list is bidirectional and ordered with respect to the column
        The column ordering follows from the insertion policy
        
        :param root: tree root
        :return list, list: data and corresponding column. Empty list if root is None
        """

        if root == NULL:
            return NULL

        cdef matrix_element_tree_s *flat_list_head, *current_element

        # Flatten lower subtree
        flat_list_head = self.subtree_to_list_flat(root.lower)

        # If no lower elements exist, the head is the current element
        if flat_list_head == NULL:
            flat_list_head = root
            root.lower = NULL

        # Else move to the tail and add the subtree root
        else:
            current_element = flat_list_head
            while current_element.higher != NULL:
                current_element = current_element.higher

            # Attach the element with the bidirectional pointers
            current_element.higher = root
            root.lower = current_element

        # Flatten higher subtree and attach it to the tail of the flat list
        root.higher = self.subtree_to_list_flat(root.higher)

        # Attach the element with the bidirectional pointers
        if root.higher != NULL:
            root.higher.lower = root

        return flat_list_head



    cdef from_linked_list_to_python_list(self, matrix_element_tree_s * head):

        data = []
        column = []

        while head != NULL:

            if head.data != 0.0:
                data.append(head.data)
                column.append(head.column)

            head = head.higher

        return column, data



    cdef subtree_free_memory(self, matrix_element_tree_s* root):
        """
        The function frees all struct in the subtree whose root is passed as a parameter, root included 
        
        :param root: tree root
        """

        if root != NULL:
            # If the root exists, open recursion
            self.subtree_free_memory(root.higher)
            self.subtree_free_memory(root.lower)

            # Once the lower elements have been reached, start freeing from the bottom
            PyMem_Free(root)



    cdef list_free_memory(self, matrix_element_tree_s * head):
        """
        The function frees all struct in the list whose head is passed as a parameter, head included 
        
        :param head: list head
        """

        if head != NULL:
            # If the root exists, open recursion
            self.subtree_free_memory(head.higher)

            # Once the tail element have been reached, start freeing from them
            PyMem_Free(head)



    cdef matrix_element_tree_s* build_tree_from_list_flat(self, matrix_element_tree_s* flat_list_head):
        """
        The function builds a tree containing the passed data. This is the recursive function, the 
        data should be sorted by te caller
        To ensure the tree is balanced, data is sorted according to the column   
        
        :param row: row in which to create new tree
        :param column_vector: column coordinates 
        :param data_vector: cell data
        """

        if flat_list_head == NULL:
            return NULL


        cdef long list_length = 0
        cdef long middle_element_step = 0

        cdef matrix_element_tree_s *current_element, *middleElement, *tree_root

        current_element = flat_list_head
        middleElement = flat_list_head

        # Explore the flat list moving the middle elment every tho jumps
        while current_element != NULL:
            current_element = current_element.higher
            list_length += 1
            middle_element_step += 1

            if middle_element_step == 2:
                middleElement = middleElement.higher
                middle_element_step = 0

        tree_root = middleElement

        # To execute the recursion it is necessary to cut the flat list
        # The last of the lower elements will have to be a tail
        if middleElement.lower != NULL:
            middleElement.lower.higher = NULL

            tree_root.lower = self.build_tree_from_list_flat(flat_list_head)


        # The first of the higher elements will have to be a head
        if middleElement.higher != NULL:
            middleElement.higher.lower = NULL

            tree_root.higher = self.build_tree_from_list_flat(middleElement.higher)


        return tree_root




    cdef matrix_element_tree_s* topK_selection_from_list(self, matrix_element_tree_s* head, long TopK):
        """
        The function selects the topK highest elements in the given list 
        
        :param head: head of the list
        :param TopK: number of highest elements to preserve
        :return matrix_element_tree_s*: head of the new list
        """

        cdef head_pointer_tree_s *vector_pointer_to_list_elements
        cdef matrix_element_tree_s *current_element
        cdef long list_length, index, selected_count

        # Get list size
        current_element = head
        list_length = 0

        while current_element != NULL:
            list_length += 1
            current_element = current_element.higher


        # If list elements are not enough to perform a selection, return
        if list_length < TopK:
            return head

        # Allocate vector that will be used for sorting
        vector_pointer_to_list_elements = < head_pointer_tree_s *> PyMem_Malloc(list_length * sizeof(head_pointer_tree_s))

        # Fill vector wit pointers to list elements
        current_element = head
        for index in range(list_length):
            vector_pointer_to_list_elements[index].head = current_element
            current_element = current_element.higher


        # Sort array elements on their data field
        qsort(vector_pointer_to_list_elements, list_length, sizeof(head_pointer_tree_s), compare_struct_on_data)

        # Sort only the TopK according to their column field
        # Sort is from lower to higher, therefore the elements to be considered are from len-topK to len
        qsort(&vector_pointer_to_list_elements[list_length-TopK], TopK, sizeof(head_pointer_tree_s), compare_struct_on_column)


        # Rebuild list attaching the consecutive elements
        index = list_length-TopK

        # Detach last TopK element from previous ones
        vector_pointer_to_list_elements[index].head.lower = NULL

        while index<list_length-1:
            # Rearrange bidirectional pointers
            vector_pointer_to_list_elements[index+1].head.lower = vector_pointer_to_list_elements[index].head
            vector_pointer_to_list_elements[index].head.higher = vector_pointer_to_list_elements[index+1].head

            index += 1

        # Last element in vector will be the hew head
        vector_pointer_to_list_elements[list_length - 1].head.higher = NULL

        # Get hew list head
        current_element = vector_pointer_to_list_elements[list_length-TopK].head

        # If there are exactly enough elements to reach TopK, index == 0 will be the tail
        # Else, index will be the tail and the other elements will be removed
        index = list_length - TopK - 1
        if index > 0:

            index -= 1
            while index >= 0:
                PyMem_Free(vector_pointer_to_list_elements[index].head)
                index -= 1

        # Free array
        PyMem_Free(vector_pointer_to_list_elements)


        return current_element






##################################################################################################################
#####################
#####################            TEST FUNCTIONS
#####################
##################################################################################################################


    cpdef test_list_tee_conversion(self, long row):
        """
        The function tests the inner data structure conversion from tree to C linked list and back to tree
        
        :param row: row to use for testing
        """

        cdef matrix_element_tree_s *head, *tree_root
        cdef matrix_element_tree_s *current_element, *previous_element

        head = self.subtree_to_list_flat(self.row_pointer[row].head)
        current_element = head

        cdef numElements_higher = 0
        cdef numElements_lower = 0

        while current_element != NULL:
            numElements_higher += 1
            previous_element = current_element
            current_element = current_element.higher

        current_element = previous_element
        while current_element != NULL:
            numElements_lower += 1
            current_element = current_element.lower

        assert numElements_higher == numElements_lower, 'Bidirectional linked list not consistent.' \
                                                        ' From head to tail element count is {}, from tail to head is {}'.format(
                                                        numElements_higher, numElements_lower)

        print("Bidirectional list link - Passed")

        column_original, data_original = self.from_linked_list_to_python_list(head)

        assert numElements_higher == len(column_original), \
            'Data structure size inconsistent. LinkedList is {}, Python list is {}'.format(numElements_higher, len(column_original))

        for index in range(len(column_original)-1):
            assert column_original[index] < column_original[index+1],\
                'Columns not ordered correctly. Tree not flattened properly'

        print("Bidirectional list ordering - Passed")

        # Transform list into tree and back into list, as it is easy to test
        tree_root = self.build_tree_from_list_flat(head)
        head = self.subtree_to_list_flat(tree_root)

        cdef numElements_higher_after = 0
        cdef numElements_lower_after = 0

        current_element = head

        while current_element != NULL:
            numElements_higher_after += 1
            previous_element = current_element
            current_element = current_element.higher

        current_element = previous_element
        while current_element != NULL:
            numElements_lower_after += 1
            current_element = current_element.lower

        print("Bidirectional list from tree link - Passed")

        assert numElements_higher_after == numElements_lower_after, \
            'Bidirectional linked list after tree construction not consistent. ' \
            'From head to tail element count is {}, from tail to head is {}'.format(
            numElements_higher_after, numElements_lower_after)

        assert numElements_higher == numElements_higher_after, \
            'Data structure size inconsistent. Original length is {}, after tree conversion is {}'.format(
                numElements_higher, numElements_higher_after)

        column_after_tree, data_after_tree = self.from_linked_list_to_python_list(head)

        assert len(column_original) == len(column_after_tree), \
            'Data structure size inconsistent. Original length is {}, after tree conversion is {}'.format(
                len(column_original), len(column_after_tree))

        for index in range(len(column_original)):
            assert column_original[index] == column_after_tree[index],\
                'After tree construction columns are not ordered properly'
            assert data_original[index] == data_after_tree[index],\
                'After tree construction data content is changed'

        print("Bidirectional list from tree ordering - Passed")



    cpdef test_topK_from_list_selection(self, long row, long topK):
        """
        The function tests the topK selection from list
        
        :param row: row to use for testing
        """

        cdef matrix_element_tree_s *head

        head = self.subtree_to_list_flat(self.row_pointer[row].head)

        column_original, data_original = self.from_linked_list_to_python_list(head)

        head = self.topK_selection_from_list(head, topK)

        column_topK, data_topK = self.from_linked_list_to_python_list(head)

        assert len(column_topK) == len(data_topK),\
            "TopK data and column lists have different length. Columns length is {}, data is {}".format(len(column_topK), len(data_topK))
        assert len(column_topK) <= topK,\
            "TopK extracted list is longer than desired value. Desired is {}, while list is {}".format(topK, len(column_topK))

        print("TopK extracted length - Passed")

        # Sort with respect to the content to select topK
        idx_sorted = np.argsort(data_original)
        idx_sorted = np.flip(idx_sorted, axis=0)
        top_k_idx = idx_sorted[0:topK]

        column_topK_numpy = np.array(column_original)[top_k_idx]
        data_topK_numpy = np.array(data_original)[top_k_idx]

        # Sort with respect to the column to ensure it is ordered as the tree flattened list
        idx_sorted = np.argsort(column_topK_numpy)
        column_topK_numpy = column_topK_numpy[idx_sorted]
        data_topK_numpy = data_topK_numpy[idx_sorted]


        assert len(column_topK_numpy) <= len(column_topK),\
            "TopK extracted list and numpy one have different length. Extracted list lenght is {}, while numpy is {}".format(
                len(column_topK_numpy), len(column_topK))


        for index in range(len(column_topK)):

            assert column_topK[index] == column_topK_numpy[index], \
                "TopK extracted list and numpy one have different content at index {} as column value." \
                " Extracted list lenght is {}, while numpy is {}".format(index, column_topK[index], column_topK_numpy[index])

            assert data_topK[index] == data_topK_numpy[index], \
                "TopK extracted list and numpy one have different content at index {} as data value." \
                " Extracted list lenght is {}, while numpy is {}".format(index, data_topK[index], data_topK_numpy[index])

        print("TopK extracted content - Passed")



##################################################################################################################
#####################
#####################            TRIANGULAR MATRIX
#####################
##################################################################################################################





import scipy.sparse as sps

import numpy as np
cimport numpy as np

#from libc.stdlib cimport malloc
# PyMem malloc and free are slightly faster than plain C equivalents as they optimize OS calls
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from cpython.array cimport array, clone


#################################
#################################       CLASS DECLARATION
#################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef class Triangular_Matrix:

    cdef long num_rows, num_cols
    cdef int isSymmetric

    cdef double** row_pointer





    def __init__(self, long num_rows, int isSymmetric = False):

        cdef int numRow, numCol

        self.num_rows = num_rows
        self.num_cols = num_rows
        self.isSymmetric = isSymmetric

        self.row_pointer = <double **> PyMem_Malloc(self.num_rows * sizeof(double*))



        # Initialize all rows to empty
        for numRow in range(self.num_rows):
            self.row_pointer[numRow] = < double *> PyMem_Malloc((numRow+1) * sizeof(double))

            for numCol in range(numRow+1):
                self.row_pointer[numRow][numCol] = 0.0



    def dealloc(self):
        """
        Remove all PyMalloc allocaded memory
        :return:
        """

        cdef int numRow

        # Free all rows memory
        for numRow in range(self.num_rows):
            PyMem_Free(self.row_pointer[numRow])

        PyMem_Free(self.row_pointer)




    cdef double add_value(self, long row, long col, double value):
        """
        The function adds a value to the specified cell. A new cell is created if necessary.         
        
        :param row: cell coordinates
        :param col:  cell coordinates
        :param value: value to add
        :return double: resulting cell value
        """

        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError("Cell is outside matrix. Matrix shape is ({},{}),"
                             " coordinates given are ({},{})".format(
                             self.num_rows, self.num_cols, row, col))

        elif col > row:

            if self.isSymmetric:
                self.row_pointer[col][row] += value

                return self.row_pointer[col][row]

            else:
                raise ValueError("Cell is in the upper triangular of the matrix,"
                                 " current matrix is lower triangular."
                                 " Coordinates given are ({},{})".format(row, col))
        else:

            self.row_pointer[row][col] += value

            return self.row_pointer[row][col]



    cdef double get_value(self, long row, long col):
        """
        The function returns the value of the specified cell.         
        
        :param row: cell coordinates
        :param col:  cell coordinates
        :return double: cell value
        """

        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError("Cell is outside matrix. Matrix shape is ({},{}), coordinates given are ({},{})".format(
                self.num_rows, self.num_cols, row, col))

        elif col > row:

            if self.isSymmetric:
                return self.row_pointer[col][row]

            else:
                raise ValueError("Cell is in the upper triangular of the matrix,"
                                 " current matrix is lower triangular."
                                 " Coordinates given are ({},{})".format(row, col))
        else:

            return self.row_pointer[row][col]




    cdef get_scipy_csr(self, long TopK = False):
        """
        The function returns the current sparse matrix as a scipy_csr object         
   
        :return double: scipy_csr object
        """
        cdef int terminate
        cdef long row, col, index

        cdef array[double] template_zero = array('d')
        cdef array[double] currentRowArray = clone(template_zero, self.num_cols, zero=True)

        # Declare numpy data type to use vetor indexing and simplify the topK selection code
        cdef np.ndarray[LONG_t, ndim=1] top_k_partition, top_k_partition_sorting
        cdef np.ndarray[np.float64_t, ndim=1] currentRowArray_np


        data = []
        indices = []
        indptr = []

        # Loop the rows
        for row in range(self.num_rows):

            #Always set indptr
            indptr.append(len(data))

            # Fill RowArray
            for col in range(self.num_cols):

                if col <= row:
                    currentRowArray[col] = self.row_pointer[row][col]
                else:
                    if self.isSymmetric:
                        currentRowArray[col] = self.row_pointer[col][row]
                    else:
                        currentRowArray[col] = 0.0


            if TopK:

                # Sort indices and select TopK
                # Using numpy implies some overhead, unfortunately the plain C qsort function is even slower
                #top_k_idx = np.argsort(this_item_weights) [-self.TopK:]

                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # because we avoid sorting elements we already know we don't care about
                # - Partition the data to extract the set of TopK items, this set is unsorted
                # - Sort only the TopK items, discarding the rest
                # - Get the original item index

                currentRowArray_np = - np.array(currentRowArray)
                #
                # Get the unordered set of topK items
                top_k_partition = np.argpartition(currentRowArray_np, TopK-1)[0:TopK]
                # Sort only the elements in the partition
                top_k_partition_sorting = np.argsort(currentRowArray_np[top_k_partition])
                # Get original index
                top_k_idx = top_k_partition[top_k_partition_sorting]

                for index in range(len(top_k_idx)):

                    col = top_k_idx[index]

                    if currentRowArray[col] != 0.0:
                        indices.append(col)
                        data.append(currentRowArray[col])

            else:

                for index in range(self.num_cols):

                    if currentRowArray[index] != 0.0:
                        indices.append(index)
                        data.append(currentRowArray[index])


        #Set terminal indptr
        indptr.append(len(data))

        return sps.csr_matrix((data, indices, indptr), shape=(self.num_rows, self.num_cols))



