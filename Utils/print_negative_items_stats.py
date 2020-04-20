#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/06/2019

@author: Maurizio Ferrari Dacrema
"""

import scipy.sparse as sps
import numpy as np
import traceback

def logical_iff(a, b):

    a_or_not_b = np.logical_or(a, np.logical_not(b))
    not_a_or_b = np.logical_or(np.logical_not(a), b)

    xnor = np.logical_and(a_or_not_b, not_a_or_b)

    return xnor


def logical_implies(p, q):
    return np.logical_or(np.logical_not(p), q)

def logical_implies_false_count(p, q):
    return np.logical_not(logical_implies(p, q)).sum()


def print_implication_result(string_message, p, q):

    p_true_count = np.sum(p)
    #
    # implication_flag_array = logical_implies(p, q)

    # Count how many times the implication is false
    # So, how many times P is True but Q is false
    false_count = logical_implies_false_count(p, q)

    print("{}, False for {}/{} ({:.2f} %) users".format(string_message,
                                                       false_count,
                                                       p_true_count,
                                                       false_count/p_true_count*100))




def print_iff_result(string_message, p, q):

    iff_outcome = logical_iff(p, q)

    true_count = iff_outcome.sum()
    false_count = len(iff_outcome) - true_count

    print("{}, True for {}/{} ({:.2f} %) users, False for {}".format(string_message,
                                                       true_count,
                                                       len(p),
                                                       true_count/len(p)*100,
                                                       false_count)
          )





def print_negative_items_stats(URM_train, URM_validation, URM_test, URM_test_negative):

    URM_train = URM_train.copy()
    URM_validation = URM_validation.copy()
    URM_test = URM_test.copy()
    URM_test_negative = URM_test_negative.copy()

    import traceback

    URM_test_negative_csr = sps.csr_matrix(URM_test_negative)

    user_negatives = np.ediff1d(URM_test_negative_csr.indptr)

    print("Max num negatives is {}, min num negatives is {} (nonzero is {}), users with less than max are {} of {}".format(np.max(user_negatives),
                                                                                                          np.min(user_negatives),
                                                                                                          np.min(user_negatives[user_negatives!=0]),
                                                                                                          np.sum(user_negatives!=np.max(user_negatives)),
                                                                                                          URM_test_negative_csr.shape[0]))

    from Utils.assertions_on_data_for_experiments import assert_disjoint_matrices

    remove_overlapping_data_flag = False

    print("Intersection between URM_test_negative and URM_train + URM_validation")
    try:
        assert_disjoint_matrices([URM_train + URM_validation, URM_test_negative])
    except:
        traceback.print_exc()
        remove_overlapping_data_flag = True


    print("Intersection between URM_test_negative and URM_test")
    try:
        assert_disjoint_matrices([URM_test, URM_test_negative])
    except:
        traceback.print_exc()
        remove_overlapping_data_flag = True


    if remove_overlapping_data_flag:

        print("Removing overlapping data from URM_negative")
        URM_positive = URM_train + URM_validation + URM_test
        URM_positive.data = np.ones_like(URM_positive.data)

        URM_test_negative.data = np.ones_like(URM_test_negative.data)

        # Subtract from the URM_test_negative train items
        # A - B = B - A*B

        URM_test_negative_not_positive = URM_test_negative - URM_test_negative.multiply(URM_positive)
        URM_test_negative_not_positive = sps.csr_matrix(URM_test_negative_not_positive)

        user_negatives_not_positives = np.ediff1d(URM_test_negative_not_positive.indptr)

        print("URM test negatives non overlapping with positives: Max num negatives is {}, min num negatives is {} (nonzero is {}), users with less than max are {} of {}".format(np.max(user_negatives_not_positives),
                                                                                                          np.min(user_negatives_not_positives),
                                                                                                          np.min(user_negatives_not_positives[user_negatives_not_positives!=0]),
                                                                                                          np.sum(user_negatives_not_positives!=np.max(user_negatives_not_positives)),
                                                                                                          URM_test_negative_csr.shape[0]))



    URM_train_all = URM_train + URM_validation
    URM_train_all = sps.csr_matrix(URM_train_all)
    user_train_profile = np.ediff1d(URM_train_all.indptr)

    user_test_profile = np.ediff1d(sps.csr_matrix(URM_test).indptr)


    assert np.array_equal(logical_iff(np.array([False,  False,  True,   True]),
                                      np.array([False,  True,   False,  True])),
                                      np.array([True,   False,  False,  True]))


    print_iff_result("User presence in train data IFF presence in test", user_train_profile>0, user_test_profile>0)
    print_iff_result("User presence in test data IFF presence in negative items test", user_test_profile>0, user_negatives>0)
    print_iff_result("User presence in train data IFF presence in negative items test", user_train_profile>0, user_negatives>0)
