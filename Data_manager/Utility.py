#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Simone Boglio
"""

import scipy.sparse as sps
import numpy as np

def filter_urm(urm, user_min_number_ratings=1, item_min_number_ratings=1):
    # keep only users with at least n ratings, same for the items
    # NOTE: this operation re index both users and items (we get a more compact URM)
    urm = sps.csr_matrix(urm)
    urm.eliminate_zeros()
    users_to_select_mask = np.ediff1d(urm.indptr) >= user_min_number_ratings
    urm = urm[users_to_select_mask, :]
    urm = sps.csc_matrix(urm)
    items_to_select_mask = np.ediff1d(urm.indptr) >= item_min_number_ratings
    urm = urm[:, items_to_select_mask]
    return urm.tocsr()


def print_stat_urm(urm, title=''):
    if title!='': title = '{:10}'.format(title)+'-> '
    n_users = urm.shape[0]
    n_items = urm.shape[1]
    n_ratings = urm.data.shape[0]
    density = n_ratings / (n_users * n_items) * 100
    print('{}users: {} \titems: {} \tratings: {:8d} \tdensity: {:.3f}%'.format(title,n_users, n_items, n_ratings, density))

def print_stat_icm(urm, title=''):
    if title!='': title = '{:10}'.format(title)+'-> '
    n_users = urm.shape[0]
    n_items = urm.shape[1]
    n_ratings = urm.data.shape[0]
    density = n_ratings / (n_users * n_items) * 100
    print('{}items: {} \tfeatures: {} \tvalues: {:9d} \tdensity: {:.3f}%'.format(title,n_users, n_items, n_ratings, density))

def print_stat_ucm(urm, title=''):
    if title!='': title = '{:10}'.format(title)+'-> '
    n_users = urm.shape[0]
    n_items = urm.shape[1]
    n_ratings = urm.data.shape[0]
    density = n_ratings / (n_users * n_items) * 100
    print('{}users: {} \tfeatures: {} \tvalues: {:9d} \tdensity: {:.3f}%'.format(title,n_users, n_items, n_ratings, density))


def print_stat_datareader(datareader):

    URM_train = datareader.URM_DICT["URM_train"]
    URM_validation = datareader.URM_DICT["URM_validation"]
    URM_test = datareader.URM_DICT["URM_test"]

    print_stat_urm(URM_train + URM_test + URM_validation, title='DATASET')

    for URM_name, URM_object in datareader.URM_DICT.items():
        print_stat_urm(URM_object, title=URM_name)


    if len(datareader.ICM_DICT)>0:
        for ICM_name, ICM_object in datareader.ICM_DICT.items():
            print_stat_icm(ICM_object, title=ICM_name)


    if hasattr(datareader, 'UCM'):
        # Rimpiazzarlo con UCM_DICT ?
        print_stat_ucm(datareader.UCM, title='UCM')
