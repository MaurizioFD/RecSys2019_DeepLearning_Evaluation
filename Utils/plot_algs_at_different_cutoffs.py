#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/11/18

@author: Maurizio Ferrari Dacrema
"""


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import itertools, os


def plot_algs_at_different_cutoffs(evaluator, folder_path,
                                   recommender1_object, recommender1_name,
                                   recommender_dict_name_to_object):

    """

    :param evaluator:
    :param cutoff_list:
    :param folder_path:
    :param recommender_dict_name_to_object: Contains a dictionary [alg_lable]-> recommender_object
    :return:
    """

    # If directory does not exist, create
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



    assert recommender1_name not in recommender_dict_name_to_object, "This function requires the recommender1_name NOT to be in recommender_dict_name_to_object"

    recommender_dict_name_to_object[recommender1_name] = recommender1_object


    recommender_dict_name_to_result_dict = {}
    metric_list = None
    cutoff_list = None


    for recommender_name in recommender_dict_name_to_object.keys():

        recommender_object = recommender_dict_name_to_object[recommender_name]

        results_dict, results_dict_string = evaluator.evaluateRecommender(recommender_object)

        logFile = open(folder_path + recommender_name + "_all_cutoffs.txt", "a")
        logFile.write(results_dict_string)
        logFile.close()

        print("Results for {}: \n{}".format(recommender_name, results_dict_string))

        recommender_dict_name_to_result_dict[recommender_name] = results_dict

        if metric_list is None:
            cutoff_list = list(results_dict.keys())
            metric_list = list(results_dict[cutoff_list[0]].keys())






    import pickle

    pickle.dump(recommender_dict_name_to_result_dict,
                open(folder_path + "recommender_dict_name_to_result_dict", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)




    for recommender2_name in recommender_dict_name_to_object.keys():

        if recommender2_name == recommender1_name:
            continue

        results_dict_recommender1 = recommender_dict_name_to_result_dict[recommender1_name]
        results_dict_recommender2 = recommender_dict_name_to_result_dict[recommender2_name]


        for metric_to_plot in metric_list:

            results_dict_recommender1_x_values = [results_dict_recommender1[cutoff][metric_to_plot] for cutoff in cutoff_list]
            results_dict_recommender2_x_values = [results_dict_recommender2[cutoff][metric_to_plot] for cutoff in cutoff_list]



            # Turn interactive plotting off
            plt.ioff()

            # Ensure it works even on SSH
            plt.switch_backend('agg')

            plt.xlabel('Cutoff value')
            plt.ylabel("{} value".format(metric_to_plot))
            plt.title("Metric value for different cutoff values")

            x_tick = cutoff_list

            marker_list = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]
            marker_iterator_local = itertools.cycle(marker_list)


            plt.plot(x_tick, results_dict_recommender1_x_values, linewidth=3, label = recommender1_name,
                     linestyle = "-", marker = marker_iterator_local.__next__())

            plt.plot(x_tick, results_dict_recommender2_x_values, linewidth=3, label = recommender2_name,
                     linestyle = "-", marker = marker_iterator_local.__next__())

            plt.legend()

            plt.savefig(folder_path + "Metric value for different cutoff values_{}_{}_{}".format(recommender1_name, recommender2_name, metric_to_plot))

            plt.close()





