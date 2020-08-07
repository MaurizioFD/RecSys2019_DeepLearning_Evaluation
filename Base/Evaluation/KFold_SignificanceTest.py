#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/07/18

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
from scipy import stats
#
# import random
# from statsmodels.sandbox.stats.multicomp import multipletests
#
# # as example, all null hypotheses are true
# pvals = [random.random() for _ in range(10)]
# is_reject, corrected_pvals, _, _ = multipletests(pvals, alpha=0.1, method='fdr_bh')


def write_log(string, log_file = None):

    print(string)

    if log_file is not None:
        log_file.write(string + "\n")
        log_file.flush()




import traceback

import pandas as pd

def is_distribution_normal(data_list, alpha_input):

    """
    Shapiro-Wilk Test of Normality if less than 50 participants; the Kolmogorov-Smirnov Test if equal or more than 50
    Use the Shapiro-Wilk test first and look at the Kolmogorov Smirnov test afterwards because it is generally more sensitive.

    :param data_list:
    :param alpha_input:
    :return:
    """

    """
    null hypothesis that the data was drawn from a normal distribution.
    
    """

    result_dict = {}

    try:
        stat, p_value = stats.shapiro(data_list)
        stat, p_value = float(stat), float(p_value)
        result_dict["normal_saphiro_p_value"] = p_value
        result_dict["normal_saphiro_pass"] = p_value>=alpha_input
    except:
        traceback.print_exc()
        result_dict["normal_saphiro_p_value"] = None
        result_dict["normal_saphiro_pass"] = None

    if len(data_list)>=8:
        try:
            # D’Agostino and Pearson’s test
            stat, p_value = stats.normaltest(data_list)
            stat, p_value = float(stat), float(p_value)
            result_dict["normal_dagostino_p_value"] = p_value
            result_dict["normal_dagostino_pass"] = p_value>=alpha_input
        except:
            traceback.print_exc()
            # result_dict["normal_dagostino_p_value"] = None
            # result_dict["normal_dagostino_pass"] = None


    try:
        # Kolmogorov Smirnov test
        # Null hypothesis is that the data comes from a normal distribution of given mean and std
        # KS Test can detect the variance and is more sensitive then others
        mean, std = np.mean(data_list), np.std(data_list)
        stat, p_value = stats.kstest(data_list, cdf='norm', args=(mean, std))
        stat, p_value = float(stat), float(p_value)
        result_dict["normal_kolmogorov_p_value"] = p_value
        result_dict["normal_kolmogorov_pass"] = p_value>=alpha_input
    except:
        traceback.print_exc()
        result_dict["normal_kolmogorov_p_value"] = None
        result_dict["normal_kolmogorov_pass"] = None


    result_dict["normal_distribution_flag"] = all([result_dict[key] for key in result_dict if "_pass" in key and result_dict[key] is not None])


    return result_dict



def compute_k_fold_significance(list_1, alpha_input, other_list,  this_label = None, other_label = None, verbose = True, log_file = None):
    """
    Type 1 Errors: we identify as significant somenthing which is not, due to random chance. Lower alpha values reduce this error rate.
    Bonferroni correction is VERY conservative and also reduces the true positives rate.
    http://www.nonlinear.com/support/progenesis/comet/faq/v2.0/pq-values.aspx


    https://multithreaded.stitchfix.com/blog/2015/10/15/multiple-hypothesis-testing/
    https://www.scipy-lectures.org/packages/statistics/index.html


    If Data Is Gaussian:
        Use Parametric Statistical Methods
    Else:
        Use Nonparametric Statistical Methods


    :param list_1:
    :param alpha:
    :param other_lists:
    :return:
    """

    result_dict = {}

    this_label = this_label if this_label is not None else "List 1"
    other_label = other_label if other_label is not None else "List 2"

    result_dict[this_label + " mean"], result_dict[this_label + " std"] = float(np.mean(list_1)), float(np.std(list_1))
    result_dict[other_label + " mean"], result_dict[other_label + " std"] = float(np.mean(other_list)), float(np.std(other_list))

    format_string = "4E" if np.abs(np.mean(list_1))>100 or np.abs(np.mean(other_list))>100 else "4f"

    is_all_finite = np.all(np.isfinite(list_1)) and np.all(np.isfinite(other_list))
    if not is_all_finite:
        if verbose:
            write_log("{}: has {} non-finite values".format(this_label, np.sum(np.logical_not(np.isfinite(list_1)))), log_file=log_file)
            write_log("{}: has {} non-finite values".format(other_label, np.sum(np.logical_not(np.isfinite(other_list)))), log_file=log_file)
            write_log("Skipping test due to non-finite values\n", log_file=log_file)
            return result_dict



    if verbose:
        write_log("{}: {:.{format_string}} ± {:.{format_string}}".format(this_label, np.mean(list_1), np.std(list_1), format_string = format_string), log_file=log_file)
    #
    # if len(other_lists) > 1:
    #     alpha_threshold = alpha_input/len(other_lists)
    #     if verbose:
    #         write_log("Applying Bonferroni correction for {} lists, original alpha is {}, corrected alpha is {}".format(len(other_lists), alpha_input, alpha_threshold), log_file=log_file)

    assert isinstance(other_list, list) or isinstance(other_list, np.ndarray), "The provided lists must be either Python lists or numpy.ndarray"
    assert len(list_1) == len(other_list), "The provided lists have different length, list 1: {}, list 2: {}".format(len(list_1), len(other_list))

    ###
    ### Check normal distribution
    ###
    normal_result_dict_1 = is_distribution_normal(list_1, alpha_input)
    normal_result_dict_1 = {"{} {}".format(this_label, key):value for (key, value) in normal_result_dict_1.items()}
    result_dict = {**result_dict, **normal_result_dict_1}

    normal_result_dict_other = is_distribution_normal(other_list, alpha_input)
    normal_result_dict_other = {"{} {}".format(other_label, key):value for (key, value) in normal_result_dict_other.items()}
    result_dict = {**result_dict, **normal_result_dict_other}

    ###
    ### If normal, paired t-test, if not Wilcoxon Signed Rank Test
    ###
    is_normal = result_dict[this_label + " normal_distribution_flag"] and result_dict[other_label + " normal_distribution_flag"]

    ###
    ### Check if list are equals
    ###
    if np.all(np.equal(np.zeros_like(list_1), np.array(list_1) - np.array(other_list))):
        result_dict["difference_is_significant_test"] = "Lists are equal"
        statistic, p_value = np.nan, np.nan

    else:

        if is_normal:
            # Test difference between two observations of the same "individual" or data with a paired test
            # Equivalent to test whether (list_1 - other_list) has an average of 0
            statistic, p_value = stats.ttest_rel(list_1, other_list)
            result_dict["difference_is_significant_test"] = "Paired t-test"
        else:
            statistic, p_value = stats.wilcoxon(list_1, other_list)
            result_dict["difference_is_significant_test"] = "Wilcoxon signed-rank"


    statistic, p_value = float(statistic), float(p_value)
    result_dict["difference_is_significant_statistic"] = statistic
    result_dict["difference_is_significant_p_value"] = p_value

    # Value is significant only if p_value allows
    is_significant = p_value < alpha_input
    result_dict["difference_is_significant_pass"] = is_significant

    if is_significant:
        result_dict["difference_is_significant_superior"] = this_label if np.mean(list_1)>np.mean(other_list) else other_label
    else:
        result_dict["difference_is_significant_superior"] = None



    if verbose:

        write_log("{}: {:.{format_string}} ± {:.{format_string}}".format(other_label, np.mean(other_list), np.std(other_list), format_string = format_string),
                  log_file=log_file)

        write_log("{}, using {}: statistic {:.4f}, p_value: {:.4f}, alpha: {:.4f}. {} {}\n".format(
                    "IS normal" if is_normal else "NOT normal",
                    result_dict["difference_is_significant_test"],
                    statistic,
                    p_value,
                    alpha_input,
                    "IS significant." if result_dict["difference_is_significant_pass"] else "NOT significant.",
                    "Higher is {}.".format(result_dict["difference_is_significant_superior"]) if result_dict["difference_is_significant_superior"] is not None else ""),
                log_file=log_file)


    return result_dict













class KFold_SignificanceTest(object):
    """KFold_SignificanceTest"""

    def __init__(self, n_folds, log_label = None, allow_overwrite = False, log_file = None):
        super(KFold_SignificanceTest, self).__init__()

        assert n_folds>0, "KFold_SignificanceTest: n_folds cannot be negative"

        self._result_list = [None]*n_folds
        self._n_folds = n_folds
        self._log_file = log_file
        self._log_label = log_label
        self._allow_overwrite = allow_overwrite


    def set_results_in_fold(self, fold_index, result_dict):

        if self._result_list[fold_index] is not None and not self._allow_overwrite:
            raise Exception("KFold_SignificanceTest: set_results_in_fold {} would overwrite previously set value".format(fold_index))

        self._result_list[fold_index] = result_dict.copy()


    def get_results(self):
        return self._result_list.copy()

    def get_fold_number(self):
        return self._n_folds

    def _get_label(self):
        return self._log_label

    def run_significance_test(self, other_result_repository, metric = None, alpha = 0.05, verbose = True, dataframe_path = None):

        assert isinstance(other_result_repository, KFold_SignificanceTest), "KFold_SignificanceTest: run_significance_test must receive another repository as parameter"
        assert other_result_repository.get_fold_number()== self.get_fold_number(), "KFold_SignificanceTest: run_significance_test other repository must have the same number of folds"

        result_list_other = other_result_repository.get_results()
        other_label = other_result_repository._get_label()

        if metric is None:
            metric_list = list(result_list_other[0].keys())
        else:
            metric_list = [metric]


        result_df = None

        for metric in metric_list:

            if verbose:
                write_log("Significance test on metric: {}".format(metric), log_file = self._log_file)

            list_this = []
            list_other = []

            for fold_index in range(self._n_folds):

                list_this.append(self._result_list[fold_index][metric])
                list_other.append(result_list_other[fold_index][metric])

            result_dict = compute_k_fold_significance(list_this, alpha, list_other, this_label = self._log_label, other_label = other_label, verbose = verbose, log_file=self._log_file)
            # result_dict["metric"] = metric

            if result_df is None:
                df_columns = ["difference_is_significant_pass", "difference_is_significant_superior"]
                df_columns.extend([key for key in result_dict.keys() if "difference" in key and  key not in df_columns])
                df_columns.extend([key for key in result_dict.keys() if key not in df_columns])
                result_df = pd.DataFrame(columns=df_columns, index=metric_list)

            # result_df = result_df.append(result_dict, ignore_index=True)
            result_df.loc[metric] = result_dict

        if dataframe_path is not None:
            result_df.to_csv(dataframe_path)


        return result_df