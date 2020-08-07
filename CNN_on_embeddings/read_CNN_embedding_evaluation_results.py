#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29/11/19

@author: Maurizio Ferrari Dacrema
"""


from Base.Evaluation.KFold_SignificanceTest import KFold_SignificanceTest
from Base.DataIO import DataIO
from scipy import stats
import numpy as np





def get_metric_value_list(result_list, cutoff, metric):

    metric_value_list = [None]*len(result_list)

    for index in range(len(metric_value_list)):

        metric_value = result_list[index][str(cutoff)][metric]
        metric_value_list[index] = metric_value

    return metric_value_list



def read_permutation_results(output_folder_path, n_permutations,
                                cutoff, metrics, file_result_name_root,
                                convolution_model_name, pretrained_model_name, pretrained_model_class, experiment_type = "fit_ablation"):

    log_file_main_diagonal = open(output_folder_path + file_result_name_root + "_significance_main_diagonal.txt", "w", encoding='utf-8')
    log_file_off_diagonal = open(output_folder_path + file_result_name_root + "_significance_off_diagonal.txt", "w", encoding='utf-8')

    result_repo_full_map = KFold_SignificanceTest(n_permutations, log_label="Full map")

    result_repo_main_diagonal = KFold_SignificanceTest(n_permutations, log_label="Main diagonal", log_file=log_file_main_diagonal)
    result_repo_off_diagonal = KFold_SignificanceTest(n_permutations, log_label="Off diagonal", log_file=log_file_off_diagonal)


    permutation_full_map_result_list = [None]*n_permutations
    permutation_main_diag_result_list = [None]*n_permutations
    permutation_off_diag_result_list = [None]*n_permutations

    ############ PERMUTATIONS
    for permutation_index in range(n_permutations):

        dataIO = DataIO(folder_path = output_folder_path + "{}_all_map/all_map_{}/".format(experiment_type, permutation_index))
        search_metadata = dataIO.load_data(convolution_model_name + "_metadata")

        permutation_full_map_result_list[permutation_index] = search_metadata["result_on_last"]
        result_repo_full_map.set_results_in_fold(permutation_index, search_metadata["result_on_last"][str(cutoff)])


        dataIO = DataIO(folder_path = output_folder_path + "{}_main_diagonal/main_diagonal_{}/".format(experiment_type, permutation_index))
        search_metadata = dataIO.load_data(convolution_model_name + "_metadata")

        permutation_main_diag_result_list[permutation_index] = search_metadata["result_on_last"]
        result_repo_main_diagonal.set_results_in_fold(permutation_index, search_metadata["result_on_last"][str(cutoff)])


        dataIO = DataIO(folder_path = output_folder_path + "{}_off_diagonal/off_diagonal_{}/".format(experiment_type, permutation_index))
        search_metadata = dataIO.load_data(convolution_model_name + "_metadata")

        permutation_off_diag_result_list[permutation_index] = search_metadata["result_on_last"]
        result_repo_off_diagonal.set_results_in_fold(permutation_index, search_metadata["result_on_last"][str(cutoff)])

    dataframe_main_diagonal = result_repo_main_diagonal.run_significance_test(result_repo_full_map,
                                               dataframe_path = output_folder_path + file_result_name_root + "_significance_main_diagonal.csv")
    dataframe_off_diagonal = result_repo_off_diagonal.run_significance_test(result_repo_full_map,
                                               dataframe_path = output_folder_path + file_result_name_root + "_significance_off_diagonal.csv")



    output_file = open(output_folder_path + file_result_name_root + "_table.txt", "w")

    output_file.write("\t&" + "\t&".join(metric for metric in metrics) + "\\\\ \n")

    if pretrained_model_name is not None:
        try:
            dataIO = DataIO(folder_path = output_folder_path + pretrained_model_name + "/")
            search_metadata = dataIO.load_data(pretrained_model_class.RECOMMENDER_NAME + "_metadata")

            BPRMF_result = search_metadata["result_on_test_best"]

            output_file.write(pretrained_model_name + " \t&" + "\t&".join("{:.4f}".format(BPRMF_result[str(cutoff)][metric]) for metric in metrics) + "\\\\ \n")
        except:
            pass

    output_file.write("{} full \t&".format(convolution_model_name))

    for metric in metrics:
        esperiment_result_list_metric = get_metric_value_list(permutation_full_map_result_list, cutoff, metric)
        output_file.write("{:.4f} $\pm$ {:.4f}\t{}".format(np.mean(esperiment_result_list_metric), np.std(esperiment_result_list_metric), "&" if metric!=metrics[-1] else ""))



    output_file.write("\\\\ \n")


    output_file.write("{} main diag \t&".format(convolution_model_name))

    for metric in metrics:
        esperiment_result_list_metric = get_metric_value_list(permutation_main_diag_result_list, cutoff, metric)
        output_file.write("{:.4f} $\pm$ {:.4f}\t{}".format(np.mean(esperiment_result_list_metric), np.std(esperiment_result_list_metric), "&" if metric!=metrics[-1] else ""))

    output_file.write("\\\\ \n")

    output_file.write("Is significant\t&")

    for metric in metrics:
        is_significant = dataframe_main_diagonal.loc[metric]["difference_is_significant_pass"]
        output_file.write("{}\t{}".format(is_significant, "&" if metric != metrics[-1] else ""))


    output_file.write("\\\\ \n")
    output_file.write("\\midrule \n")

    output_file.write("{} off diag \t&".format(convolution_model_name))

    for metric in metrics:
        esperiment_result_list_metric = get_metric_value_list(permutation_off_diag_result_list, cutoff, metric)
        output_file.write("{:.4f} $\pm$ {:.4f}\t{}".format(np.mean(esperiment_result_list_metric), np.std(esperiment_result_list_metric), "&" if metric!=metrics[-1] else ""))

    output_file.write("\\\\ \n")
    # output_file.write("Paired t-test p value\t&")
    output_file.write("Is significant\t&")

    for metric in metrics:
        # esperiment_result_list_metric = get_metric_value_list(permutation_full_map_result_list, cutoff, metric)
        # permutation_diag_result_list_metric = get_metric_value_list(permutation_off_diag_result_list, cutoff, metric)
        #
        # t_statistic, p_value = stats.ttest_rel(esperiment_result_list_metric, permutation_diag_result_list_metric)
        #
        # output_file.write("{:.3f}\t{}".format(p_value, "&" if metric != metrics[-1] else ""))
        is_significant = dataframe_off_diagonal.loc[metric]["difference_is_significant_pass"]
        output_file.write("{}\t{}".format(is_significant, "&" if metric != metrics[-1] else ""))

    output_file.write("\\\\ \n")
    output_file.close()



