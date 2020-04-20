'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import os.path
from time import time
from numpy import linalg as LA


class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path, recent_len=5):
        '''
        Constructor
        '''
        # if not testing:
        self.recent_len=recent_len
        print("Loading trainMatrix...")
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.num_users, self.num_items = self.trainMatrix.shape
        # print("Loading trainMatrixRecent...")
        # self.trainMatrixRecent = self.load_recent_rating_file_as_matrix(path + ".train.rating")
        print("Loading testRatings...")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        print("Loading validRatings...")
        self.validRatings = self.load_rating_file_as_list(path + ".valid.rating")
        print("Loading testNegatives...")
        try:
            self.testNegatives = self.load_negative_file(path + ".test.negative")
        except:
            try:
                self.testNegatives = self.load_negative_file(path + ".test.negative-500")
            except:
                self.testNegatives = self.load_negative_file(path + ".test.negative-100")
        assert len(self.testRatings) == len(self.testNegatives)


    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        # self.num_users = num_users+1
        # self.num_items = num_items+1
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                mat[user, item] = 1.0
                line = f.readline()
        return mat

    def load_recent_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        '''
        # Get number of users and items
        num_users, num_items = self.num_users, self.num_items
        # Construct matrix
        mat_recent = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        last_user = -1
        recent_count = 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                if user==last_user:
                    recent_count+=1
                else:
                    last_user=user
                    recent_count=1
                if recent_count<=self.recent_len:
                    mat_recent[user, item] = 1.0
                line = f.readline()
        assert mat_recent.nnz == self.recent_len*num_users
        return mat_recent


    def find_item_neighbors(self, num_neighbors):
        if (os.path.exists("similar_matrix_id_%d.npy" % num_neighbors) and
                os.path.exists("similar_matrix_weight_%d.npy" % num_neighbors)):
            similar_matrix_id = np.load("similar_matrix_id_%d.npy" % num_neighbors)
            similar_matrix_weight = np.load("similar_matrix_weight_%d.npy" % num_neighbors)
        else:
            t1 = time()
            similar_matrix_id = np.zeros(shape=(self.num_items, num_neighbors), dtype=int) - 1
            similar_matrix_weight = np.zeros(shape=(self.num_items, num_neighbors), dtype=float) - 1
            rating = self.trainMatrix.toarray().transpose()
            for j in range(rating.shape[0]):
                if rating[j].sum() == 0:
                    rating[j] += 1. / self.num_items
            for i in range(self.num_items):
                item_rating = rating[i]
                print ("item %d .... [%.1f s]" % (i, time() - t1))
                weight = (item_rating * rating).sum(axis=1) / LA.norm(rating, ord=2, axis=1) / \
                         LA.norm(item_rating, ord=2)
                topk_id = np.argpartition(weight, -num_neighbors - 1)[-num_neighbors - 1:]
                topk_weight = weight[topk_id]
                sorted_topk_id = topk_id[np.argsort(topk_weight)]
                sorted_topk_weight = np.sort(topk_weight)
                similar_matrix_id[i] = sorted_topk_id[:-1]
                # softmax over item neighbors' weights
                similar_matrix_weight[i] = np.exp(sorted_topk_weight[:-1]) / np.exp(sorted_topk_weight[:-1]).sum()
                # similar_matrix_weight[i] = sorted_topk_weight[:-1]
                # t1=time()
                # for j in xrange(self.num_items):
                #     if j==i:
                #         continue
                #     else:
                #         weight = self.get_similarity(item_rating, self.trainMatrix[:,j].toarray())
                #     if weight < similar_matrix_weight[i].min():
                #         continue
                #     else:
                #         # update_similar_matrix
                #         min_index = similar_matrix_weight[i].argmin()
                #         similar_matrix_weight[i, min_index] = weight
                #         similar_matrix_id[i, min_index] = j
                if i % (self.num_items // 20) == 0:
                    print ("%d/%% items done..." % (i // (self.num_items // 20) * 5))
                if i % 10 == 0:
                    np.save("similar_matrix_id_%d" % num_neighbors, similar_matrix_id)
                    np.save("similar_matrix_weight_%d" % num_neighbors, similar_matrix_weight)
            print ("Minimum id in matrix is %d" % similar_matrix_id.min())
            print ("Minimum weight in matrix is %.2f" % similar_matrix_weight.min())
            np.save("similar_matrix_id_%d" % num_neighbors, similar_matrix_id)
            np.save("similar_matrix_weight_%d" % num_neighbors, similar_matrix_weight)
        return [similar_matrix_id, similar_matrix_weight]

        # def get_similarity(self, num_neighbors):
        #     #using cosine similarity
        #
        #     weight = (a.transpose().dot(b).sum())/(a.sum()*b.sum())
        #     return weight
