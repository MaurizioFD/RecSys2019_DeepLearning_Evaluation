#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/02/19

@author: Maurizio Ferrari Dacrema
"""

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use("pgf")
import matplotlib.pyplot as plt

import itertools
import numpy as np
import scipy.sparse as sps



def plot_popularity_bias(URM_object_list, URM_name_list, output_img_path, sort_on_all = False):

    shape = URM_object_list[0].shape

    for URM_object in URM_object_list:
        assert URM_object.shape == shape

    n_items = shape[1]



    # Turn interactive plotting off
    plt.ioff()

    # Ensure it works even on SSH
    plt.switch_backend('agg')

    marker_list = ['o', 's', '^', 'v', 'D']
    marker_iterator_local = itertools.cycle(marker_list)


    plt.xlabel('Item id')
    plt.ylabel("Normalized number of interactions per item")
    #plt.title("Item popularity distribution for different URM splits")

    x_tick = np.arange(0,n_items, dtype=np.int)

    item_sorted_indices = None

    if sort_on_all:

        URM_all = None

        for URM_object in URM_object_list:
            if URM_all is None:
                URM_all = URM_object.copy()
            else:
                URM_all += URM_object.copy()

        URM_object = sps.csc_matrix(URM_all)

        item_popularity = np.ediff1d(URM_object.indptr)

        if item_sorted_indices is None:
            item_sorted_indices = np.argsort(-item_popularity)


    for URM_index in range(len(URM_name_list)):

        URM_object = URM_object_list[URM_index]
        URM_name = URM_name_list[URM_index]

        URM_object = sps.csc_matrix(URM_object)

        item_popularity = np.ediff1d(URM_object.indptr)

        if item_sorted_indices is None:
            item_sorted_indices = np.argsort(-item_popularity)

        max_popularity = item_popularity.max()

        plt.plot(x_tick, item_popularity[item_sorted_indices]/max_popularity, linewidth=1, label = URM_name,
                 linestyle = "-", marker = marker_iterator_local.__next__(), markersize=3, zorder=-URM_index)

    plt.legend()

    plt.grid(False, which= 'major')
    # fig = plt.figure()
    # fig.patch.set_facecolor('white')
    # plt.rcParams['figure.facecolor'] = 'white'

    plt.savefig(output_img_path + ".png", dpi = 1200, bbox_inches='tight')

    plt.savefig(output_img_path + ".pdf", dpi = 1200, bbox_inches='tight')

    # plt.savefig(output_img_path + ".pgf", bbox_inches='tight')

    plt.close()




from Base.Evaluation.metrics import Gini_Diversity, Diversity_Herfindahl, Shannon_Entropy
from scipy.stats import kendalltau


def Gini_Index(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = np.array(array, dtype=np.float)
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


def save_popularity_statistics(URM_object_list, URM_name_list, output_file_path):

    shape = URM_object_list[0].shape

    for URM_object in URM_object_list:
        assert URM_object.shape == shape

    n_items = shape[1]
    ignore_items = np.array([])

    latex_file = open(output_file_path, "w")
    latex_file.write("\tMax pop \t&Min pop \t&Avg pop \t&Gini Index \t&Kendall Tau \t&Shannon \t&Herfindahl \\\\ \n")

    base_item_popularity = None


    for URM_index, URM_name in enumerate(URM_name_list):

        URM_object = URM_object_list[URM_index]
        URM_object = sps.csc_matrix(URM_object)

        URM_string = "{}\t&".format(URM_name)

        item_popularity = np.ediff1d(URM_object.indptr)

        max_pop = item_popularity.max()
        min_pop = item_popularity.min()
        avg_pop = item_popularity.mean()

        gini_index = gini(item_popularity)

        if base_item_popularity is None:
            base_item_popularity = item_popularity.copy()


        kendall_tau, p_value = kendalltau(base_item_popularity, item_popularity)


        values_list = [max_pop, min_pop, avg_pop, gini_index, kendall_tau]



        metric_object_dict = {
            # "Gini Diversity": Gini_Diversity(n_items, ignore_items),
            "Shannon Entropy": Shannon_Entropy(n_items, ignore_items),
            "Herfindahl Index": Diversity_Herfindahl(n_items, ignore_items),
        }



        for metric_name, metric_object in metric_object_dict.items():

            metric_object.recommended_counter = item_popularity.copy()
            metric_value = metric_object.get_metric_value()

            values_list.append(metric_value)


        URM_string += "\t&".join("{:.2f}".format(el) for el in values_list)

        latex_file.write(URM_string + "\\\\ \n")





