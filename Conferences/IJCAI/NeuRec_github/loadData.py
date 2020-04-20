import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


def load_movielens_ineurec(path="data/1m_ratings.dat", test_size=0.2, header=['user_id', 'item_id', 'rating'],
                           sep="\t"):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    print(df.user_id.unique().shape[0])
    print(df.item_id.unique().shape[0])
    n_users = df.user_id.max()#.unique().shape[0])
    n_items = df.item_id.max()#.unique().shape[0])

    train_data, test_data = train_test_split(df, test_size=0.1)
    train_data, test_data = train_test_split(train_data, test_size=0.2)

    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []
    train_rating_1 = []

    train_dict = {}
    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        r = line[3]
        train_dict[(u, i)] = r

    count = 0
    for u in range(n_users):
        for i in range(n_items):
            train_row.append(u)
            train_col.append(i)
            if (u, i) in train_dict.keys():
                count = count + 1
                train_rating.append(1)
                train_rating_1.append(1)
            else:
                train_rating.append(0)
                train_rating_1.append(0)

    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))
    train_matrix_item = csr_matrix((train_rating_1, (train_col, train_row)), shape=(n_items, n_users))
    all_items = set(np.arange(n_items))

    neg_user_item_matrix = {}
    train_user_item_matrix = []
    for u in range(n_users):
        neg_user_item_matrix[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))

    for i in range(n_items):
        train_user_item_matrix.append(list(train_matrix_item.getrow(i).toarray()[0]))

    test_row = []
    test_col = []
    test_rating = []
    unique_users = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        unique_users.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
    test_user_item_matrix = {}
    for u in range(n_users):
        test_user_item_matrix[u] = test_matrix.getrow(u).nonzero()[1]

    num_training = n_users * n_items

    return train_matrix.todok(), train_matrix_item.todok(), train_user_item_matrix, neg_user_item_matrix, test_matrix.todok(), test_user_item_matrix, n_users, n_items, num_training, set(
        unique_users)


def load_movielens_uneurec(path="data/u.data", test_size=0.2, header=['user_id', 'item_id', 'rating', 'time'],
                           sep="\t"):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    print(df.user_id.unique().shape[0])
    print(df.item_id.unique().shape[0])
    n_users = df.user_id.max()#.unique().shape[0])
    n_items = df.item_id.max()#.unique().shape[0])

    train_data, test_data = train_test_split(df, test_size=0.1)
    train_data, test_data = train_test_split(train_data, test_size=0.2)

    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []
    train_rating_1 = []

    train_dict = {}
    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        r = line[3]
        train_dict[(u, i)] = r

    count = 0
    for u in range(n_users):
        for i in range(n_items):
            train_row.append(u)
            train_col.append(i)
            if (u, i) in train_dict.keys():
                count = count + 1
                train_rating.append(1)
                train_rating_1.append(1)
            else:
                train_rating.append(0)
                train_rating_1.append(0)

    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))
    train_matrix_item = csr_matrix((train_rating_1, (train_col, train_row)), shape=(n_items, n_users))
    all_items = set(np.arange(n_items))

    neg_user_item_matrix = {}
    train_user_item_matrix = []
    for u in range(n_users):
        neg_user_item_matrix[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))
        train_user_item_matrix.append(list(train_matrix.getrow(u).toarray()[0]))

    test_row = []
    test_col = []
    test_rating = []
    unique_users = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        unique_users.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
    test_user_item_matrix = {}
    for u in range(n_users):
        test_user_item_matrix[u] = test_matrix.getrow(u).nonzero()[1]

    num_training = n_users * n_items

    return train_matrix.todok(), train_matrix_item.todok(), train_user_item_matrix, neg_user_item_matrix, test_matrix.todok(), test_user_item_matrix, n_users, n_items, num_training, set(
        unique_users)
