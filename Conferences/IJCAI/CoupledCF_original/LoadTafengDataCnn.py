import scipy.sparse as sp
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler


def load_rating_file_as_list(path='tafeng/'):
    ratingList = []
    with open(path+'test.rating', "r", encoding="utf-8") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList


def load_negative_file(path='tafeng/'):
    negativeList = []
    with open(path+'test.negative', "r", encoding="utf-8") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1:]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList


def load_rating_train_as_matrix(path='tafeng/'):
    # Get number of users and items
    num_users = 32266
    num_items = 23812

    # Construct matrix
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open(path+"train.rating", "r", encoding="utf-8") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            mat[user, item] = 1.0
            line = f.readline()
    return mat


def load_itemGenres_as_matrix(path='tafeng/', split=False):
    data = open(path+"item.data", encoding="utf-8").readlines()
    '''
        0. item_id
        1. original_id
        2. sub_class
        3. Amount
        4. Asset
        5. price    
    '''
    sub_class, asset, price = [], [], []
    for row in data[1:]:
        row = row.strip('\n')
        row = row.split('\t')
        sub_class.append(row[2])  # categorical
        asset.append(float(row[4]))  # numerical
        price.append(float(row[5]))  # numerical

    set_sub_class = set(sub_class)
    dict_sub_class = {k: i + 1 for i, k in enumerate(set_sub_class)}
    sub_class = [dict_sub_class[l] for l in sub_class]

    # transform to darray
    sub_class = np.asarray(sub_class)
    asset = np.asarray(asset)
    price = np.asarray(price)

    # transform to one-hot
    # sub_class = to_categorical(sub_class)

    # reshape to (n,1)
    sub_class = sub_class.reshape(-1, 1)
    asset = asset.reshape(-1, 1)
    price = price.reshape(-1, 1)

    # hstack the numerical cols to a matrix
    numerical_cols = np.hstack((asset, price))

    # norm to [0-1]
    min_max_scaler = MinMaxScaler()
    numerical_cols = min_max_scaler.fit_transform(numerical_cols)

    # hstack the item_id, sub_class,asset,price to a matrix
    item_attrs_mat = np.hstack((sub_class, numerical_cols))
    first_row = [0, 0, 0]
    item_attrs_mat = np.row_stack([first_row, item_attrs_mat])

    num_items = 23812

    return num_items, item_attrs_mat


def load_user_attributes(path='tafeng/'):
    '''
    0.  user_id
    1.  original_id
    2.  age ---catgorical
    3.  region ---catgorical
    '''
    age_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11}
    region_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
    age, region = [], []

    data = open(path+"user.data", encoding="utf-8").readlines()
    for row in data[1:]:
        row = row.strip('\n')
        row = row.replace(" ", "")
        arr = row.split("\t")
        age.append(age_dict[arr[2]])
        region.append(region_dict[arr[3]])

    # transform to darray
    age = np.asarray(age).reshape(-1, 1)
    region = np.asarray(region).reshape(-1, 1)

    # transform to one-hot
    age = to_categorical(age)
    region = to_categorical(region)

    # hstack the attributes to a matrix
    user_attrs_mat = np.hstack((age, region))
    first_row = np.zeros((1,21))
    user_attrs_mat = np.row_stack([first_row, user_attrs_mat])

    num_users = 32266

    return num_users, user_attrs_mat
