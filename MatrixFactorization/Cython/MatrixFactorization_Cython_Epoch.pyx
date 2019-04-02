"""
Created on 07/09/17

@author: Anonymous authors
"""

#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from Base.Recommender_utils import check_matrix
import numpy as np
cimport numpy as np
import time
import sys

from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, RAND_MAX


cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item


cdef struct MSE_sample:
    long user
    long item
    double rating




cdef class MatrixFactorization_Cython_Epoch:

    cdef int n_users, n_items, n_factors
    cdef int numPositiveIteractions
    cdef algorithm_name

    cdef float learning_rate, user_reg, positive_reg, negative_reg

    cdef int batch_size

    cdef int algorithm_is_funk_svd, algorithm_is_asy_svd, algorithm_is_BPR

    cdef int[:] URM_train_indices, URM_train_indptr, profile_length
    cdef double[:] URM_train_data

    cdef double[:,:] USER_factors, ITEM_factors, ITEM_factors_Y


    # Adaptive gradient
    cdef int useAdaGrad, useRmsprop, useAdam, verbose

    cdef double [:] sgd_cache_I, sgd_cache_U
    cdef double gamma

    cdef double [:] sgd_cache_I_momentum_1, sgd_cache_I_momentum_2
    cdef double [:] sgd_cache_U_momentum_1, sgd_cache_U_momentum_2
    cdef double beta_1, beta_2, beta_1_power_t, beta_2_power_t
    cdef double momentum_1, momentum_2


    def __init__(self, URM_train, n_factors = 10, algorithm_name = "FUNK_SVD",
                 learning_rate = 0.01, user_reg = 0.0, positive_reg = 0.0, negative_reg = 0.0,
                 verbose = False,
                 batch_size = 1, sgd_mode='sgd', gamma=0.995, beta_1=0.9, beta_2=0.999):

        super(MatrixFactorization_Cython_Epoch, self).__init__()


        URM_train = check_matrix(URM_train, 'csr')

        self.numPositiveIteractions = int(URM_train.nnz * 1)
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.profile_length = np.ediff1d(URM_train.indptr)
        self.n_factors = n_factors
        self.verbose = verbose
        self.algorithm_name = algorithm_name

        self.URM_train_indices = URM_train.indices
        self.URM_train_data = np.array(URM_train.data, dtype=np.float64)
        self.URM_train_indptr = URM_train.indptr


        self.algorithm_is_funk_svd = False
        self.algorithm_is_asy_svd = False
        self.algorithm_is_BPR = False


        if self.algorithm_name == "FUNK_SVD":
            self.algorithm_is_funk_svd = True
            # W and H cannot be initialized as zero, otherwise the gradient will always be zero
            self.USER_factors = np.random.random((self.n_users, self.n_factors))
            self.ITEM_factors = np.random.random((self.n_items, self.n_factors))

        elif self.algorithm_name == "ASY_SVD":
            self.algorithm_is_asy_svd = True
            # W and H cannot be initialized as zero, otherwise the gradient will always be zero
            self.USER_factors = np.random.random((self.n_items, self.n_factors))
            self.ITEM_factors = np.random.random((self.n_items, self.n_factors))

        elif self.algorithm_name == "MF_BPR":
            self.algorithm_is_BPR = True
            # W and H cannot be initialized as zero, otherwise the gradient will always be zero
            self.USER_factors = np.random.random((self.n_users, self.n_factors))
            self.ITEM_factors = np.random.random((self.n_items, self.n_factors))

        else:
            raise ValueError("Loss value not recognized")





        self.useAdaGrad = False
        self.useRmsprop = False
        self.useAdam = False


        if sgd_mode=='adagrad':
            self.useAdaGrad = True
            self.sgd_cache_I = np.zeros((self.ITEM_factors.shape[0]), dtype=np.float64)
            self.sgd_cache_U = np.zeros((self.USER_factors.shape[0]), dtype=np.float64)

        elif sgd_mode=='rmsprop':
            self.useRmsprop = True
            self.sgd_cache_I = np.zeros((self.ITEM_factors.shape[0]), dtype=np.float64)
            self.sgd_cache_U = np.zeros((self.USER_factors.shape[0]), dtype=np.float64)

            # Gamma default value suggested by Hinton
            # self.gamma = 0.9
            self.gamma = gamma

        elif sgd_mode=='adam':
            self.useAdam = True
            self.sgd_cache_I_momentum_1 = np.zeros((self.ITEM_factors.shape[0]), dtype=np.float64)
            self.sgd_cache_I_momentum_2 = np.zeros((self.ITEM_factors.shape[0]), dtype=np.float64)

            self.sgd_cache_U_momentum_1 = np.zeros((self.USER_factors.shape[0]), dtype=np.float64)
            self.sgd_cache_U_momentum_2 = np.zeros((self.USER_factors.shape[0]), dtype=np.float64)

            # Default value suggested by the original paper
            # beta_1=0.9, beta_2=0.999
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.beta_1_power_t = beta_1
            self.beta_2_power_t = beta_2

        elif sgd_mode=='sgd':
            pass
        else:
            raise ValueError(
                "SGD_mode not valid. Acceptable values are: 'sgd', 'adagrad', 'rmsprop', 'adam'. Provided value was '{}'".format(
                    sgd_mode))



        self.learning_rate = learning_rate
        self.user_reg = user_reg
        self.positive_reg = positive_reg
        self.negative_reg = negative_reg


        if batch_size!=1:
            print("MiniBatch not implemented, reverting to default value 1")
        self.batch_size = 1


    # Using memoryview instead of the sparse matrix itself allows for much faster access
    cdef int[:] getSeenItems(self, long index):
        return self.URM_train_indices[self.URM_train_indptr[index]:self.URM_train_indptr[index + 1]]




    def epochIteration_Cython(self):

        if self.algorithm_is_funk_svd:
            self.epochIteration_Cython_FUNK_SVD_SGD()

        elif self.algorithm_is_asy_svd:
            self.epochIteration_Cython_ASY_SVD_SGD()

        elif self.algorithm_is_BPR:
            self.epochIteration_Cython_BPR_SGD()




    def epochIteration_Cython_FUNK_SVD_SGD(self):

        # Get number of available interactions
        cdef long num_total_batch = int(len(self.URM_train_data) / self.batch_size) + 1

        cdef MSE_sample sample
        cdef long factor_index, num_current_batch, processed_samples_last_print, print_block_size = 500
        cdef double prediction, gradient, adaptive_gradient_item, adaptive_gradient_user

        cdef int numSeenItems


        cdef double H_i, W_u, cumulative_loss = 0.0


        cdef long start_time_epoch = time.time()
        cdef long last_print_time = start_time_epoch

        for num_current_batch in range(num_total_batch):

            # Uniform user sampling with replacement
            sample = self.sampleMSE_Cython()

            prediction = 0.0

            for factor_index in range(self.n_factors):
                prediction += self.USER_factors[sample.user, factor_index] * self.ITEM_factors[sample.item, factor_index]

            gradient = sample.rating - prediction
            cumulative_loss += gradient**2

            adaptive_gradient_item = self.adaptive_gradient_item(gradient, sample.item)
            adaptive_gradient_user = self.adaptive_gradient_user(gradient, sample.user)


            for factor_index in range(self.n_factors):

                # Copy original value to avoid messing up the updates
                H_i = self.ITEM_factors[sample.item, factor_index]
                W_u = self.USER_factors[sample.user, factor_index]

                self.USER_factors[sample.user, factor_index] += self.learning_rate * (adaptive_gradient_user * H_i - self.user_reg * W_u)
                self.ITEM_factors[sample.item, factor_index] += self.learning_rate * (adaptive_gradient_item * W_u - self.positive_reg * H_i)




            if self.verbose and (processed_samples_last_print >= print_block_size or num_current_batch == num_total_batch-1):

                current_time = time.time()

                # Set block size to the number of items necessary in order to print every 30 seconds
                samples_per_sec = num_current_batch/(time.time() - start_time_epoch)

                print_block_size = int(samples_per_sec*30)

                if current_time - last_print_time > 30 or num_current_batch == num_total_batch-1:

                    print("{}: Processed {} ( {:.2f}% ) in {:.2f} seconds. MSE loss {:.2E}. Sample per second: {:.0f}".format(
                        self.algorithm_name,
                        num_current_batch*self.batch_size,
                        100.0* num_current_batch/num_total_batch,
                        time.time() - last_print_time,
                        cumulative_loss/(num_current_batch*self.batch_size + 1),
                        float(num_current_batch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                    last_print_time = current_time
                    processed_samples_last_print = 0

                    sys.stdout.flush()
                    sys.stderr.flush()





    def epochIteration_Cython_ASY_SVD_SGD(self):

        # Get number of available interactions
        cdef long num_total_batch = int(len(self.URM_train_data) / self.batch_size) + 1

        cdef MSE_sample sample
        cdef long num_current_batch, processed_samples_last_print, print_block_size = 500
        cdef double prediction, gradient, adaptive_gradient_item, adaptive_gradient_user

        cdef int numSeenItems

        cdef double[:] user_factors_accumulated = np.zeros(self.n_factors, dtype=np.float64)
        cdef long start_pos_seen_items, end_pos_seen_items, item_id, factor_index, item_index, user_index

        cdef double H_i, W_u, cumulative_loss = 0.0, denominator


        cdef long start_time_epoch = time.time()
        cdef long last_print_time = start_time_epoch


        for num_current_batch in range(num_total_batch):

            # Uniform user sampling with replacement
            sample = self.sampleMSE_Cython()

            prediction = 0.0

            for factor_index in range(self.n_factors):
                user_factors_accumulated[factor_index] = 0.0


            # Accumulate latent factors of rated items
            start_pos_seen_items = self.URM_train_indptr[sample.user]
            end_pos_seen_items = self.URM_train_indptr[sample.user+1]

            for item_index in range(start_pos_seen_items, end_pos_seen_items):
                item_id = self.URM_train_indices[item_index]

                for factor_index in range(self.n_factors):
                    user_factors_accumulated[factor_index] += self.USER_factors[item_id, factor_index]


            denominator = sqrt(self.profile_length[sample.user])


            for factor_index in range(self.n_factors):
                user_factors_accumulated[factor_index] /= denominator

            # Compute prediction
            for factor_index in range(self.n_factors):
                prediction += user_factors_accumulated[factor_index] * self.ITEM_factors[sample.item, factor_index]


            gradient = sample.rating - prediction
            cumulative_loss += gradient**2

            adaptive_gradient_item = self.adaptive_gradient_item(gradient, sample.item)
            adaptive_gradient_user = self.adaptive_gradient_user(gradient, sample.item)


            # Update USER factors, therefore all item factors for seen items
            for item_index in range(start_pos_seen_items, end_pos_seen_items):
                item_id = self.URM_train_indices[item_index]

                for factor_index in range(self.n_factors):

                    H_i = self.ITEM_factors[sample.item, factor_index]
                    W_u = self.USER_factors[item_id, factor_index]

                    self.USER_factors[item_id, factor_index] += self.learning_rate * (adaptive_gradient_user * H_i - self.user_reg * W_u)


            # Update ITEM factors
            for factor_index in range(self.n_factors):

                # Copy original value to avoid messing up the updates
                H_i = self.ITEM_factors[sample.item, factor_index]
                W_u = user_factors_accumulated[factor_index]

                self.ITEM_factors[sample.item, factor_index] += self.learning_rate * (adaptive_gradient_item * W_u - self.positive_reg * H_i)




            if self.verbose and (processed_samples_last_print >= print_block_size or num_current_batch == num_total_batch-1):

                current_time = time.time()

                # Set block size to the number of items necessary in order to print every 30 seconds
                samples_per_sec = num_current_batch/(time.time() - start_time_epoch)

                print_block_size = int(samples_per_sec*30)

                if current_time - last_print_time > 30 or num_current_batch == num_total_batch-1:

                    print("{}: Processed {} ( {:.2f}% ) in {:.2f} seconds. MSE loss {:.2E}. Sample per second: {:.0f}".format(
                        self.algorithm_name,
                        num_current_batch*self.batch_size,
                        100.0* num_current_batch/num_total_batch,
                        time.time() - last_print_time,
                        cumulative_loss/(num_current_batch*self.batch_size + 1),
                        float(num_current_batch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                    last_print_time = current_time
                    processed_samples_last_print = 0

                    sys.stdout.flush()
                    sys.stderr.flush()



    def epochIteration_Cython_BPR_SGD(self):

        # Get number of available interactions
        cdef long num_total_batch = int(self.n_users / self.batch_size) + 1


        cdef BPR_sample sample
        cdef long u, i, j
        cdef long factor_index, num_current_batch, processed_samples_last_print, print_block_size = 500
        cdef double x_uij, sigmoid_user, sigmoid_item

        cdef int numSeenItems


        cdef double H_i, H_j, W_u, cumulative_loss = 0.0


        cdef long start_time_epoch = time.time()
        cdef long last_print_time = start_time_epoch

        for num_current_batch in range(num_total_batch):

            # Uniform user sampling with replacement
            sample = self.sampleBPR_Cython()

            u = sample.user
            i = sample.pos_item
            j = sample.neg_item

            x_uij = 0.0

            for factor_index in range(self.n_factors):
                x_uij += self.USER_factors[u,factor_index] * (self.ITEM_factors[i,factor_index] - self.ITEM_factors[j,factor_index])

            # Use gradient of log(sigm(-x_uij))
            sigmoid_item = 1 / (1 + exp(x_uij))
            sigmoid_user = sigmoid_item

            cumulative_loss += x_uij**2

            sigmoid_item_i = self.adaptive_gradient_item(sigmoid_item, i)
            sigmoid_item_j = self.adaptive_gradient_item(sigmoid_item, j)

            sigmoid_user = self.adaptive_gradient_user(sigmoid_user, u)




            for factor_index in range(self.n_factors):

                # Copy original value to avoid messing up the updates
                H_i = self.ITEM_factors[i, factor_index]
                H_j = self.ITEM_factors[j, factor_index]
                W_u = self.USER_factors[u, factor_index]

                self.USER_factors[u, factor_index] += self.learning_rate * (sigmoid_user * ( H_i - H_j ) - self.user_reg * W_u)
                self.ITEM_factors[i, factor_index] += self.learning_rate * (sigmoid_item_i * ( W_u ) - self.positive_reg * H_i)
                self.ITEM_factors[j, factor_index] += self.learning_rate * (sigmoid_item_j * (-W_u ) - self.negative_reg * H_j)




            if self.verbose and (processed_samples_last_print >= print_block_size or num_current_batch == num_total_batch-1):

                current_time = time.time()

                # Set block size to the number of items necessary in order to print every 30 seconds
                samples_per_sec = num_current_batch/(time.time() - start_time_epoch)

                print_block_size = int(samples_per_sec*30)

                if current_time - last_print_time > 30 or num_current_batch == num_total_batch-1:

                    print("{}: Processed {} ( {:.2f}% ) in {:.2f} seconds. BPR loss {:.2E}. Sample per second: {:.0f}".format(
                        self.algorithm_name,
                        num_current_batch*self.batch_size,
                        100.0* num_current_batch/num_total_batch,
                        time.time() - last_print_time,
                        cumulative_loss/(num_current_batch*self.batch_size + 1),
                        float(num_current_batch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                    last_print_time = current_time
                    processed_samples_last_print = 0

                    sys.stdout.flush()
                    sys.stderr.flush()









    def get_USER_factors(self):
        return np.array(self.USER_factors)


    def get_ITEM_factors(self):
        return np.array(self.ITEM_factors)



    cdef double adaptive_gradient_item(self, double gradient, long item_id):


        cdef double gradient_update


        if self.useAdaGrad:
            self.sgd_cache_I[item_id] += gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache_I[item_id]) + 1e-8)


        elif self.useRmsprop:
            self.sgd_cache_I[item_id] = self.sgd_cache_I[item_id] * self.gamma + (1 - self.gamma) * gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache_I[item_id]) + 1e-8)


        elif self.useAdam:

            self.sgd_cache_I_momentum_1[item_id] = \
                self.sgd_cache_I_momentum_1[item_id] * self.beta_1 + (1 - self.beta_1) * gradient

            self.sgd_cache_I_momentum_2[item_id] = \
                self.sgd_cache_I_momentum_2[item_id] * self.beta_2 + (1 - self.beta_2) * gradient**2


            self.momentum_1 = self.sgd_cache_I_momentum_1[item_id]/ (1 - self.beta_1_power_t)
            self.momentum_2 = self.sgd_cache_I_momentum_2[item_id]/ (1 - self.beta_2_power_t)

            gradient_update = self.momentum_1/ (sqrt(self.momentum_2) + 1e-8)


        else:

            gradient_update = gradient


        return gradient_update



    cdef double adaptive_gradient_user(self, double gradient, long user_id):


        cdef double gradient_update

        if self.useAdaGrad:
            self.sgd_cache_U[user_id] += gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache_U[user_id]) + 1e-8)


        elif self.useRmsprop:
            self.sgd_cache_U[user_id] = self.sgd_cache_U[user_id] * self.gamma + (1 - self.gamma) * gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache_U[user_id]) + 1e-8)


        elif self.useAdam:

            self.sgd_cache_U_momentum_1[user_id] = \
                self.sgd_cache_U_momentum_1[user_id] * self.beta_1 + (1 - self.beta_1) * gradient

            self.sgd_cache_U_momentum_2[user_id] = \
                self.sgd_cache_U_momentum_2[user_id] * self.beta_2 + (1 - self.beta_2) * gradient**2


            self.momentum_1 = self.sgd_cache_U_momentum_1[user_id]/ (1 - self.beta_1_power_t)
            self.momentum_2 = self.sgd_cache_U_momentum_2[user_id]/ (1 - self.beta_2_power_t)

            gradient_update = self.momentum_1/ (sqrt(self.momentum_2) + 1e-8)


        else:

            gradient_update = gradient


        return gradient_update






    cdef MSE_sample sampleMSE_Cython(self):

        cdef MSE_sample sample = MSE_sample(-1,-1,-1.0)
        cdef long index, start_pos_seen_items, end_pos_seen_items

        cdef int numSeenItems = 0

        # Skip users with no interactions or with no negative items
        while numSeenItems == 0 or numSeenItems == self.n_items:

            sample.user = rand() % self.n_users

            start_pos_seen_items = self.URM_train_indptr[sample.user]
            end_pos_seen_items = self.URM_train_indptr[sample.user+1]

            numSeenItems = end_pos_seen_items - start_pos_seen_items


        index = rand() % numSeenItems

        sample.item = self.URM_train_indices[start_pos_seen_items + index]
        sample.rating = self.URM_train_data[start_pos_seen_items + index]

        return sample




    cdef BPR_sample sampleBPR_Cython(self):

        cdef BPR_sample sample = BPR_sample(-1,-1,-1)
        cdef long index, start_pos_seen_items, end_pos_seen_items

        cdef int negItemSelected, numSeenItems = 0


        # Skip users with no interactions or with no negative items
        # Skip users with no interactions or with no negative items
        while numSeenItems == 0 or numSeenItems == self.n_items:

            sample.user = rand() % self.n_users

            start_pos_seen_items = self.URM_train_indptr[sample.user]
            end_pos_seen_items = self.URM_train_indptr[sample.user+1]

            numSeenItems = end_pos_seen_items - start_pos_seen_items
            

        index = rand() % numSeenItems

        sample.pos_item = self.URM_train_indices[start_pos_seen_items + index]



        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        # for every user
        while (not negItemSelected):

            sample.neg_item = rand() % self.n_items

            index = 0
            while index < numSeenItems and self.URM_train_indices[start_pos_seen_items + index]!=sample.neg_item:
                index+=1

            if index == numSeenItems:
                negItemSelected = True


        return sample
