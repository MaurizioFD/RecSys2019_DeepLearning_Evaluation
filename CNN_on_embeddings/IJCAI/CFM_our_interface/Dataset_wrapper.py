'''
Wrapper between LoadData and the URM

@author: Federico Parroni
'''

import numpy as np
import os
from shutil import copyfile
import scipy.sparse as sps
from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip


def dict_to_sparse_matrix(d:dict, shape:tuple):
    """ Convert a dictionary to a sparse matrix """
    tot_items = 0
    row_ind = []
    col_ind = []
    for k, v in d.items():
        c = len(v)
        row_ind.extend([k] * c)
        col_ind.extend(v)
        tot_items += c
    data = np.ones(tot_items)

    sparse_matrix = sps.csr_matrix((data, (row_ind, col_ind)), shape=shape)

    assert sparse_matrix.nnz == len(data), "Duplicate entries found"

    return sparse_matrix



class Interaction:
    """ Stores an interaction loaded from a csv file with format: <user_features>, <item_features>. """

    def __init__(self, line):
        parsed_line = line.strip().split(',')
        self.user_features_str = parsed_line[0]
        self.item_features_str = parsed_line[1]
        self.user_features = list(map(int, self.user_features_str.split('-')))
        self.item_features = list(map(int, self.item_features_str.split('-')))


class LoadData(object):
    '''given the path of data, return the data format for CFM for Top-N recommendation
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X_user' and 'X_item' refers to features for context-aware user and item
    Test_data: same as Train_data
    '''

    # Two files are needed in the path
    def __init__(self, path, dataset):
        self.path = path + dataset + "/"
        self.trainfile = self.path + "train.csv"
        self.valfile = self.path + "validation.csv"
        self.testfile = self.path + "test.csv"

        use_validation = os.path.isfile(self.valfile)

        parsed_train = self.parse_file(self.trainfile)
        parsed_valid = self.parse_file(self.valfile) if use_validation else None
        parsed_test = self.parse_file(self.testfile)

        # get the max user and item feature id
        self.user_field_M, self.item_field_M = self.get_length(parsed_train, parsed_valid, parsed_test)
        print("user_field_M", self.user_field_M)
        print("item_field_M", self.item_field_M)
        print("field_M", self.user_field_M + self.item_field_M)

        self.item_bind_M = self.bind_item(parsed_train, parsed_valid, parsed_test)  # assaign a userID for a specific user-context
        self.user_bind_M = self.bind_user(parsed_train, parsed_valid, parsed_test)  # assaign a itemID for a specific item-feature
        print("item_bind_M", len(self.binded_items.values()))
        print("user_bind_M", len(self.binded_users.values()))
        # for data reader compatibility
        self.shape = (self.user_bind_M, self.item_bind_M)
        self.data = np.array([])

        # URM_train coincide with self.user_positive_list dictionary
        self.user_positive_list = self.get_positive_list(parsed_train)  # userID positive itemID
        self.Train_data, self.Valid_data, self.Test_data = self.construct_data(parsed_train, parsed_valid, parsed_test)

        self.user_positive_list_valid = self.get_positive_list(parsed_valid) if use_validation else None
        self.user_positive_list_test = self.get_positive_list(parsed_test)


    def parse_file(self, filepath):
        """ Parse the file (train or test) and load in memory for future usage. Return a list of Interaction """
        parsed_data = []
        with open(filepath) as f:
            lines = f.readlines()
            for line in lines:
                parsed_data.append(Interaction(line))
        return parsed_data

    def get_length(self, parsed_train, parsed_valid, parsed_test):
        '''
        map the user fields in all files, kept in self.user_fields dictionary
        :return:
        '''
        length_user = 0
        length_item = 0
        for line in parsed_train:
            for user_feature in line.user_features:
                if user_feature > length_user:
                    length_user = user_feature
            for item_feature in line.item_features:
                if item_feature > length_item:
                    length_item = item_feature

        if parsed_valid is not None:
            for line in parsed_valid:
                for user_feature in line.user_features:
                    if user_feature > length_user:
                        length_user = user_feature
                for item_feature in line.item_features:
                    if item_feature > length_item:
                        length_item = item_feature

        for line in parsed_test:
            for user_feature in line.user_features:
                if user_feature > length_user:
                    length_user = user_feature
            for item_feature in line.item_features:
                if item_feature > length_item:
                    length_item = item_feature

        return length_user + 1, length_item + 1

    def bind_item(self, parsed_train, parsed_valid, parsed_test):
        '''
        Bind item and feature (associate an item to an iid: "2-16057" -> iid)
        '''
        self.binded_items = {}  # dic{feature: id}
        self.item_map = {}      # dic{id: feature}

        # build the list of all items features, this is cached to to speed up scores
        # computation. See get_scores_per_user() method of FM
        self.all_items_features_list = []

        self.bind_i(parsed_train)
        if parsed_valid is not None:
            self.bind_i(parsed_valid)
        self.bind_i(parsed_test)
        return len(self.binded_items)

    def bind_i(self, interactions):
        '''
        Read a feature file and bind
        :param file: feature file
        '''
        i = len(self.binded_items)
        for line in interactions:
            item_features = line.item_features_str
            if item_features not in self.binded_items:
                self.binded_items[item_features] = i
                self.item_map[i] = item_features
                self.all_items_features_list.append(line.item_features)
                i = i + 1

    def bind_user(self, parsed_train, parsed_valid, parsed_test):
        '''
        Map the item fields in all files, kept in self.item_fields dictionary
        (associate a user to an uid: "1-1000" -> uid)
        '''
        self.binded_users = {}
        self.user_map = {}      # dic{id: feature}
        self.bind_u(parsed_train)
        if parsed_valid is not None:
            self.bind_u(parsed_valid)
        self.bind_u(parsed_test)
        return len(self.binded_users)

    def bind_u(self, interactions):
        '''
        Read a feature file and bind
        :param file:
        :return:
        '''
        i = len(self.binded_users)
        for line in interactions:
            user_features = line.user_features_str
            if user_features not in self.binded_users:
                self.binded_users[user_features] = i
                self.user_map[i] = line.user_features
                i = i + 1

    def get_positive_list(self, interactions):
        '''
        Obtain for each user a positive item lists (associate uid -> [iid])
        :param file: train file
        :return:
        '''
        user_positive_list = {}
        for line in interactions:
            user_id = self.binded_users[line.user_features_str]
            item_id = self.binded_items[line.item_features_str]
            if user_id in user_positive_list:
                user_positive_list[user_id].append(item_id)
            else:
                user_positive_list[user_id] = [item_id]

        return user_positive_list

    def construct_data(self, parsed_train, parsed_valid, parsed_test):
        '''
        Construct train and test data
        :return:
        '''
        X_user, X_item = self.read_data(parsed_train)
        Train_data = self.construct_dataset(X_user, X_item)
        print("# of training:", len(X_user))

        if parsed_valid is not None:
            X_user, X_item = self.read_data(parsed_valid)
            Valid_data = self.construct_dataset(X_user, X_item)
            print("# of validation:", len(X_user))
        else:
            Valid_data = None

        X_user, X_item = self.read_data(parsed_test)
        Test_data = self.construct_dataset(X_user, X_item)
        print("# of test:", len(X_user))
        return Train_data, Valid_data, Test_data

    # lists of user and item
    def read_data(self, interactions):
        '''
        read raw data
        :param file: data file
        :return: structured data
        '''
        # read a data file;
        X_user = []
        X_item = []
        for line in interactions:
            X_user.append(line.user_features[0:])
            X_item.append(line.item_features[0:])

        return X_user, X_item

    def construct_dataset(self, X_user, X_item):
        '''
        Construct dataset
        :param X_user: user structured data
        :param X_item: item structured data
        :return:
        '''
        Data_Dic = {}
        indexs = range(len(X_user))
        Data_Dic['X_user'] = [X_user[i] for i in indexs]
        Data_Dic['X_item'] = [X_item[i] for i in indexs]
        return Data_Dic




""" Data Splitter """
class DataLine:
    def __init__(self, line_number, line):
        self.line_number = line_number

        line_splitted = line.strip().split(',')
        self.user_feature = line_splitted[0]
        self.item_feature = line_splitted[1]


def split_train_validation_CFM_format(train_file_path:str, random_seed:int=2019, new_folder:str=None):
    """
    Create a train-validation split from a complete file by moving the last interaction of
    each user to the validation set. Save the new train and validation files in new_folder,
    otherwise if new_folder is None save them in the same folder of the original files.
    """
    lines = []                  # contains the line
    user_items_map = {}         # key: user_features, value: list of items
    user_datalines_map = {}     # key: user_features, value: list of DataLines

    with open(train_file_path) as f:
        n = 0
        line = f.readline()
        while line:
            lines.append(line)
            dl = DataLine(n, line)

            # update user_item_map and user_datelines_map
            if dl.user_feature in user_items_map:
                user_items_map[dl.user_feature].append(dl.item_feature)
                user_datalines_map[dl.user_feature].append(dl)
            else:
                user_items_map[dl.user_feature] = [dl.item_feature]
                user_datalines_map[dl.user_feature] = [dl]

            n += 1
            line = f.readline()

    train_line_indices = []     # indices of train lines
    valid_line_indices = []     # indices of validation lines
    num_cold_users = 0          # cold users

    print('Splitting...')
    for user_feature, dl in user_datalines_map.items():
        user_profile_length = len(dl)
        if user_profile_length >= 2:
            last_item_idx = user_profile_length - 1
            for j in range(user_profile_length):
                if j != last_item_idx:
                    # put the first items in the train set
                    train_line_indices.append(dl[j].line_number)
                else:
                    # put the last item in the validation set
                    valid_line_indices.append(dl[j].line_number)
        else:
            num_cold_users += 1

    lines = np.array(lines)
    train_data = lines[train_line_indices]
    validation_data = lines[valid_line_indices]
    assert len(lines) == len(train_data) + len(validation_data) + num_cold_users

    # save train and validation lines
    direc = os.path.dirname(train_file_path) if new_folder is None else new_folder
    train_file_path = os.path.join(direc, 'train.csv')
    valid_file_path = os.path.join(direc, 'validation.csv')

    print('Saving train in {}...'.format(train_file_path))
    with open(train_file_path, 'w') as trainf:
        trainf.writelines(train_data)
    print('Saving validation in {}...'.format(valid_file_path))
    with open(valid_file_path, 'w') as validf:
        validf.writelines(validation_data)





class DatasetCFMReader:
    """ Stores an interaction loaded from a csv file with format: <user_features>, <item_features>. """

    DATASET_NAME = ""

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, data_folder, dataset_name):

        self.DATASET_NAME = dataset_name
        pre_splitted_filename = "splitted_data_"

        # If directory does not exist, create
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        data_folder_full = os.path.join(data_folder, "full")
        data_folder_validation = os.path.join(data_folder, "validation")

        if not os.path.exists(data_folder_full):
            os.makedirs(data_folder_full)

        if not os.path.exists(data_folder_validation):
            os.makedirs(data_folder_validation)

        try:
            print("Dataset_{}: Attempting to load pre-splitted data".format(self.DATASET_NAME))

            self.CFM_data_class_validation = LoadData(data_folder_validation, "")
            self.CFM_data_class_full = LoadData(data_folder_full, "")

            for attrib_name, attrib_object in load_data_dict_zip(data_folder, pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:


            print("Dataset_{}: Pre-splitted data not found, building new one".format(self.DATASET_NAME))

            original_data_path = 'CNN_on_embeddings/IJCAI/CFM_github/Data/'

            original_train_filepath = os.path.join(original_data_path, self.DATASET_NAME, 'train.csv')
            original_test_filepath = os.path.join(original_data_path, self.DATASET_NAME, 'test.csv')

            # Split train data in train-validation and copy original test data
            copyfile(original_train_filepath, os.path.join(data_folder_full, 'train.csv'))
            copyfile(original_test_filepath, os.path.join(data_folder_full, 'test.csv'))
            copyfile(original_test_filepath, os.path.join(data_folder_validation, 'test.csv'))

            split_train_validation_CFM_format(original_train_filepath, new_folder = data_folder_validation)

            self.CFM_data_class_validation = LoadData(data_folder_validation, "")
            self.CFM_data_class_full = LoadData(data_folder_full, "")

            URM_shape = (self.CFM_data_class_validation.user_bind_M, self.CFM_data_class_validation.item_bind_M)

            self.URM_DICT = {
                "URM_train_tuning_only": dict_to_sparse_matrix(self.CFM_data_class_validation.user_positive_list, shape=URM_shape),
                "URM_validation_tuning_only": dict_to_sparse_matrix(self.CFM_data_class_validation.user_positive_list_valid, shape=URM_shape),
                "URM_test_tuning_only": dict_to_sparse_matrix(self.CFM_data_class_validation.user_positive_list_test, shape=URM_shape),
                "URM_train_full": dict_to_sparse_matrix(self.CFM_data_class_full.user_positive_list, shape=URM_shape),
                "URM_test_full": dict_to_sparse_matrix(self.CFM_data_class_full.user_positive_list_test, shape=URM_shape),
            }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, data_folder, pre_splitted_filename)


        print("{}: Dataset loaded".format(self.DATASET_NAME))
