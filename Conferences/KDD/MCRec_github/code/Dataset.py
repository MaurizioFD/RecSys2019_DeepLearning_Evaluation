import scipy.sparse as sp
import numpy as np

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.types = {'u' : 1, 'm' : 2, 't' : 3, 'a' : 4, 'o' : 5}
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.num_users, self.num_items = self.trainMatrix.shape[0], self.trainMatrix.shape[1]
        
        self.user_item_map, self.item_user_map, self.train, self.item_popularity = self.load_rating_file_as_map(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        self.user_feature, self.item_feature, self.type_feature, self.age_feature, self.occ_feature = self.load_feature_as_map(path+'.bpr.user_embedding', path+'.bpr.item_embedding', path+'.bpr.type_embedding', path+'.age_fea', path+'.occ_fea')
        self.fea_size = len(self.user_feature[1])
        
        self.path_umtm, self.umtm_path_num, self.umtm_timestamp = self.load_path_as_map(path + ".umtm_5_1")
        self.path_umum, self.umum_path_num, self.umum_timestamp = self.load_path_as_map(path + '.umum_5_1')
        self.path_umtmum, self.umtmum_path_num, self.umtmum_timestamp = self.load_path_as_map(path + '.uuum_5_1')
        self.path_uuum, self.uuum_path_num, self.uuum_timestamp = self.load_path_as_map(path + '.ummm_5_1')
        assert len(self.testRatings) == len(self.testNegatives)
        
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                tmp_list = []
                for i in arr:
                    tmp_list.append(int(i))
                ratingList.append(tmp_list)
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_map(self, filename):
        user_item_map = {}
        item_user_map = {}
        train = []
        popularity_dict = {}
        max_i = 0
        total = 0
        with open(filename) as f:
            line = f.readline()
            while line != None and line != '':
                arr = line.strip().split('\t')
                u, i = int(arr[0]), int(arr[1])
                #self.num_users = max(self.num_users, u)
                max_i = max(max_i, i)
                if u not in user_item_map:
                    user_item_map[u] = {}
                if i not in item_user_map:
                    item_user_map[i] = {}
                if i not in popularity_dict:
                    popularity_dict[i] = 0
                user_item_map[u][i] = 1.0
                item_user_map[i][u] = 1.0
                popularity_dict[i] += 1
                total += 1
                train.append([u, i])
                line = f.readline()
        #self.num_users += 1
        #self.num_items += 1
        item_popularity = [0] * max_i
        for i in popularity_dict:
            item_popularity[i - 1] = int(popularity_dict[i] ** 0.5)
        return user_item_map, item_user_map, train, item_popularity

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
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        train_list = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                train_list.append([user, item])
                mat[user, item] = 1.0
                line = f.readline()    
        return mat

    def load_feature_as_map(self, user_fea_file, item_fea_file, type_fea_file, age_fea_file, occ_fea_file):
        user_feature = np.zeros((self.num_users, 64))
        item_feature = np.zeros((self.num_items, 64))
        type_feature = np.zeros((19, 64))
        age_feature = dict()
        occ_feature = dict()
        
        with open(user_fea_file) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                u = int(arr[0])
                #user_feature[u] = list()
                for j in range(len(arr[1:])):
                    user_feature[u][j] = float(arr[j + 1])

        with open(item_fea_file) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                i = int(arr[0])
                #item_feature[i] = list()
                for j in range(len(arr[1:])):
                    item_feature[i][j] = float(arr[j + 1])

        with open(type_fea_file) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                t = int(arr[0])
                #type_feature[t] = list()
                for j in range(len(arr[1:])):
                    type_feature[t][j] = float(arr[j + 1])
        '''
        with open(age_fea_file) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                a = int(arr[0])
                age_feature[a] = list()
                for f in arr[1:]:
                    age_feature[a].append(float(f))

        with open(occ_fea_file) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                o = int(arr[0])
                occ_feature[o] = list()
                for f in arr[1:]:
                    occ_feature[o].append(float(f))
        '''
        return user_feature, item_feature, type_feature, age_feature, occ_feature
    
    def load_path_as_map(self, filename):
        print (filename)
        path_dict = {}
        path_num = 0
        timestamps = 0
        length = 2
        ctn = 0
        with open(filename) as infile:
            line = infile.readline()
            while line != None and line != "":
                arr = line.split('\t')
                u, i = arr[0].split(',')
                u = int(u)
                i = int(i) 
                path_dict[(u, i)] = []
                path_num = max(int(arr[1]), path_num)
                timestamps = len(arr[2].strip().split('-'))
                line = infile.readline()
                ctn += 1
        print("{}, {}, {}, {}".format(ctn, path_num, timestamps, length))
        with open(filename) as infile:
            line = infile.readline()
            while line != None and line != "":
                arr = line.strip().split('\t')
                u, i = arr[0].split(',')
                u, i = int(u), int(i)
                path_dict[(u, i)] = []
            
                for path in arr[2:]:
                    tmp = path.split(' ')[0].split('-')
                    node_list = []
                    for node in tmp:
                        index = int(node[1:])
                        node_list.append([self.types[node[0]], index])
                    path_dict[(u, i)].append(node_list)
                line = infile.readline()
        return path_dict, path_num, timestamps


if __name__ == '__main__':
    dataset = Dataset('../data/ml-100k')
    print (dataset.user_feature)
    print (dataset.item_feature)
    print (dataset.type_feature)
