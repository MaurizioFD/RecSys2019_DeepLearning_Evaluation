#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/06/18

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
import time, sys, copy

from enum import Enum
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

from Base.Evaluation.metrics import roc_auc, precision, precision_recall_min_denominator, recall, MAP, MRR, ndcg, arhr, rmse, \
    Novelty, Coverage_Item, Metrics_Object, Coverage_User, Gini_Diversity, Shannon_Entropy, Diversity_MeanInterList, Diversity_Herfindahl, AveragePopularity


class EvaluatorMetrics(Enum):

    ROC_AUC = "ROC_AUC"
    PRECISION = "PRECISION"
    PRECISION_RECALL_MIN_DEN = "PRECISION_RECALL_MIN_DEN"
    RECALL = "RECALL"
    MAP = "MAP"
    MRR = "MRR"
    NDCG = "NDCG"
    F1 = "F1"
    HIT_RATE = "HIT_RATE"
    ARHR = "ARHR"
    RMSE = "RMSE"
    NOVELTY = "NOVELTY"
    AVERAGE_POPULARITY = "AVERAGE_POPULARITY"
    DIVERSITY_SIMILARITY = "DIVERSITY_SIMILARITY"
    DIVERSITY_MEAN_INTER_LIST = "DIVERSITY_MEAN_INTER_LIST"
    DIVERSITY_HERFINDAHL = "DIVERSITY_HERFINDAHL"
    COVERAGE_ITEM = "COVERAGE_ITEM"
    COVERAGE_USER = "COVERAGE_USER"
    DIVERSITY_GINI = "DIVERSITY_GINI"
    SHANNON_ENTROPY = "SHANNON_ENTROPY"



def create_empty_metrics_dict(n_items, n_users, URM_train, ignore_items, ignore_users, cutoff, diversity_similarity_object):

    empty_dict = {}

    # from Base.Evaluation.ResultMetric import ResultMetric
    # empty_dict = ResultMetric()

    for metric in EvaluatorMetrics:
        if metric == EvaluatorMetrics.COVERAGE_ITEM:
            empty_dict[metric.value] = Coverage_Item(n_items, ignore_items)

        elif metric == EvaluatorMetrics.DIVERSITY_GINI:
            empty_dict[metric.value] = Gini_Diversity(n_items, ignore_items)

        elif metric == EvaluatorMetrics.SHANNON_ENTROPY:
            empty_dict[metric.value] = Shannon_Entropy(n_items, ignore_items)

        elif metric == EvaluatorMetrics.COVERAGE_USER:
            empty_dict[metric.value] = Coverage_User(n_users, ignore_users)

        elif metric == EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST:
            empty_dict[metric.value] = Diversity_MeanInterList(n_items, cutoff)

        elif metric == EvaluatorMetrics.DIVERSITY_HERFINDAHL:
            empty_dict[metric.value] = Diversity_Herfindahl(n_items, ignore_items)

        elif metric == EvaluatorMetrics.NOVELTY:
            empty_dict[metric.value] = Novelty(URM_train)

        elif metric == EvaluatorMetrics.AVERAGE_POPULARITY:
            empty_dict[metric.value] = AveragePopularity(URM_train)

        elif metric == EvaluatorMetrics.MAP:
            empty_dict[metric.value] = MAP()

        elif metric == EvaluatorMetrics.MRR:
            empty_dict[metric.value] = MRR()

        elif metric == EvaluatorMetrics.DIVERSITY_SIMILARITY:
                if diversity_similarity_object is not None:
                    empty_dict[metric.value] = copy.deepcopy(diversity_similarity_object)
        else:
            empty_dict[metric.value] = 0.0

    return  empty_dict





def get_result_string(results_run, n_decimals=7):

    output_str = ""

    for cutoff in results_run.keys():

        results_run_current_cutoff = results_run[cutoff]

        output_str += "CUTOFF: {} - ".format(cutoff)

        for metric in results_run_current_cutoff.keys():
            output_str += "{}: {:.{n_decimals}f}, ".format(metric, results_run_current_cutoff[metric], n_decimals = n_decimals)

        output_str += "\n"

    return output_str



class Evaluator(object):
    """Abstract Evaluator"""

    EVALUATOR_NAME = "Evaluator_Base_Class"

    def __init__(self, URM_test_list, cutoff_list, minRatingsPerUser=1, exclude_seen=True,
                        diversity_object = None,
                        ignore_items = None,
                        ignore_users = None):

        super(Evaluator, self).__init__()



        if ignore_items is None:
            self.ignore_items_flag = False
            self.ignore_items_ID = np.array([])
        else:
            print("Ignoring {} Items".format(len(ignore_items)))
            self.ignore_items_flag = True
            self.ignore_items_ID = np.array(ignore_items)

        self.cutoff_list = cutoff_list.copy()
        self.max_cutoff = max(self.cutoff_list)

        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen

        if not isinstance(URM_test_list, list):
            self.URM_test = URM_test_list.copy()
            URM_test_list = [URM_test_list]
        else:
            raise ValueError("List of URM_test not supported")

        self.diversity_object = diversity_object

        self.n_users, self.n_items = URM_test_list[0].shape

        # Prune users with an insufficient number of ratings
        # During testing CSR is faster
        self.URM_test_list = []
        usersToEvaluate_mask = np.zeros(self.n_users, dtype=np.bool)

        for URM_test in URM_test_list:

            URM_test = sps.csr_matrix(URM_test)
            self.URM_test_list.append(URM_test)

            rows = URM_test.indptr
            numRatings = np.ediff1d(rows)
            new_mask = numRatings >= minRatingsPerUser

            usersToEvaluate_mask = np.logical_or(usersToEvaluate_mask, new_mask)

        self.usersToEvaluate = np.arange(self.n_users)[usersToEvaluate_mask]


        if ignore_users is not None:
            print("Ignoring {} Users".format(len(ignore_users)))
            self.ignore_users_ID = np.array(ignore_users)
            self.usersToEvaluate = set(self.usersToEvaluate) - set(ignore_users)
        else:
            self.ignore_users_ID = np.array([])


        self.usersToEvaluate = list(self.usersToEvaluate)





    def evaluateRecommender(self, recommender_object):
        """
        :param recommender_object: the trained recommender object, a BaseRecommender subclass
        :param URM_test_list: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """

        raise NotImplementedError("The method evaluateRecommender not implemented for this evaluator class")



    def get_user_relevant_items(self, user_id):

        assert self.URM_test.getformat() == "csr", "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in getting relevant items"

        return self.URM_test.indices[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]


    def get_user_test_ratings(self, user_id):

        assert self.URM_test.getformat() == "csr", "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in relevant items ratings"

        return self.URM_test.data[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]






class EvaluatorHoldout(Evaluator):
    """EvaluatorHoldout"""

    EVALUATOR_NAME = "EvaluatorHoldout"

    def __init__(self, URM_test_list, cutoff_list, minRatingsPerUser=1, exclude_seen=True,
                 diversity_object = None,
                 ignore_items = None,
                 ignore_users = None):


        super(EvaluatorHoldout, self).__init__(URM_test_list, cutoff_list,
                                               diversity_object = diversity_object,
                                               minRatingsPerUser=minRatingsPerUser, exclude_seen=exclude_seen,
                                               ignore_items = ignore_items, ignore_users = ignore_users)





    def _run_evaluation_on_selected_users(self, recommender_object, usersToEvaluate, block_size = None):


        if block_size is None:
            block_size = min(1000, int(1e8/self.n_items))



        start_time = time.time()
        start_time_print = time.time()


        results_dict = {}

        for cutoff in self.cutoff_list:
            results_dict[cutoff] = create_empty_metrics_dict(self.n_items, self.n_users,
                                                             recommender_object.get_URM_train(),
                                                             self.ignore_items_ID,
                                                             self.ignore_users_ID,
                                                             cutoff,
                                                             self.diversity_object)

        n_users_evaluated = 0

        # Start from -block_size to ensure it to be 0 at the first block
        user_batch_start = 0
        user_batch_end = 0

        while user_batch_start < len(self.usersToEvaluate):

            user_batch_end = user_batch_start + block_size
            user_batch_end = min(user_batch_end, len(usersToEvaluate))

            test_user_batch_array = np.array(usersToEvaluate[user_batch_start:user_batch_end])
            user_batch_start = user_batch_end

            # Compute predictions for a batch of users using vectorization, much more efficient than computing it one at a time
            recommended_items_batch_list, scores_batch = recommender_object.recommend(test_user_batch_array,
                                                                      remove_seen_flag=self.exclude_seen,
                                                                      cutoff = self.max_cutoff,
                                                                      remove_top_pop_flag=False,
                                                                      remove_CustomItems_flag=self.ignore_items_flag,
                                                                      return_scores = True
                                                                     )


            assert len(recommended_items_batch_list) == len(test_user_batch_array), "{}: recommended_items_batch_list contained recommendations for {} users, expected was {}".format(
                self.EVALUATOR_NAME, len(recommended_items_batch_list), len(test_user_batch_array))

            assert scores_batch.shape[0] == len(test_user_batch_array), "{}: scores_batch contained scores for {} users, expected was {}".format(
                self.EVALUATOR_NAME, scores_batch.shape[0], len(test_user_batch_array))

            assert scores_batch.shape[1] == self.n_items, "{}: scores_batch contained scores for {} items, expected was {}".format(
                self.EVALUATOR_NAME, scores_batch.shape[1], self.n_items)


            # Compute recommendation quality for each user in batch
            for batch_user_index in range(len(recommended_items_batch_list)):

                test_user = test_user_batch_array[batch_user_index]

                relevant_items = self.get_user_relevant_items(test_user)
                relevant_items_rating = self.get_user_test_ratings(test_user)

                all_items_predicted_ratings = scores_batch[batch_user_index]
                user_rmse = rmse(all_items_predicted_ratings, relevant_items, relevant_items_rating)

                # Being the URM CSR, the indices are the non-zero column indexes
                recommended_items = recommended_items_batch_list[batch_user_index]
                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                n_users_evaluated += 1

                for cutoff in self.cutoff_list:

                    results_current_cutoff = results_dict[cutoff]

                    is_relevant_current_cutoff = is_relevant[0:cutoff]
                    recommended_items_current_cutoff = recommended_items[0:cutoff]

                    results_current_cutoff[EvaluatorMetrics.ROC_AUC.value]              += roc_auc(is_relevant_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.PRECISION.value]            += precision(is_relevant_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.PRECISION_RECALL_MIN_DEN.value]   += precision_recall_min_denominator(is_relevant_current_cutoff, len(relevant_items))
                    results_current_cutoff[EvaluatorMetrics.RECALL.value]               += recall(is_relevant_current_cutoff, relevant_items)
                    results_current_cutoff[EvaluatorMetrics.NDCG.value]                 += ndcg(recommended_items_current_cutoff, relevant_items, relevance=self.get_user_test_ratings(test_user), at=cutoff)
                    results_current_cutoff[EvaluatorMetrics.HIT_RATE.value]             += is_relevant_current_cutoff.sum()
                    results_current_cutoff[EvaluatorMetrics.ARHR.value]                 += arhr(is_relevant_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.RMSE.value]                 += user_rmse

                    results_current_cutoff[EvaluatorMetrics.MRR.value].add_recommendations(is_relevant_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.MAP.value].add_recommendations(is_relevant_current_cutoff, relevant_items)
                    results_current_cutoff[EvaluatorMetrics.NOVELTY.value].add_recommendations(recommended_items_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.AVERAGE_POPULARITY.value].add_recommendations(recommended_items_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.DIVERSITY_GINI.value].add_recommendations(recommended_items_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.SHANNON_ENTROPY.value].add_recommendations(recommended_items_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.COVERAGE_ITEM.value].add_recommendations(recommended_items_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.COVERAGE_USER.value].add_recommendations(recommended_items_current_cutoff, test_user)
                    results_current_cutoff[EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST.value].add_recommendations(recommended_items_current_cutoff)
                    results_current_cutoff[EvaluatorMetrics.DIVERSITY_HERFINDAHL.value].add_recommendations(recommended_items_current_cutoff)

                    if EvaluatorMetrics.DIVERSITY_SIMILARITY.value in results_current_cutoff:
                        results_current_cutoff[EvaluatorMetrics.DIVERSITY_SIMILARITY.value].add_recommendations(recommended_items_current_cutoff)


                if time.time() - start_time_print > 30 or n_users_evaluated==len(self.usersToEvaluate):

                    elapsed_time = time.time()-start_time
                    new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

                    print("{}: Processed {} ( {:.2f}% ) in {:.2f} {}. Users per second: {:.0f}".format(
                                  self.EVALUATOR_NAME,
                                  n_users_evaluated,
                                  100.0* float(n_users_evaluated)/len(self.usersToEvaluate),
                                  new_time_value, new_time_unit,
                                  float(n_users_evaluated)/elapsed_time))

                    sys.stdout.flush()
                    sys.stderr.flush()

                    start_time_print = time.time()



        return results_dict, n_users_evaluated




    def evaluateRecommender(self, recommender_object):
        """
        :param recommender_object: the trained recommender object, a BaseRecommender subclass
        :param URM_test_list: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)



        results_dict, n_users_evaluated = self._run_evaluation_on_selected_users(recommender_object, self.usersToEvaluate)


        if (n_users_evaluated > 0):

            for cutoff in self.cutoff_list:

                results_current_cutoff = results_dict[cutoff]

                for key in results_current_cutoff.keys():

                    value = results_current_cutoff[key]

                    if isinstance(value, Metrics_Object):
                        results_current_cutoff[key] = value.get_metric_value()
                    else:
                        results_current_cutoff[key] = value/n_users_evaluated

                precision_ = results_current_cutoff[EvaluatorMetrics.PRECISION.value]
                recall_ = results_current_cutoff[EvaluatorMetrics.RECALL.value]

                if precision_ + recall_ != 0:
                    # F1 micro averaged: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.8244&rep=rep1&type=pdf
                    results_current_cutoff[EvaluatorMetrics.F1.value] = 2 * (precision_ * recall_) / (precision_ + recall_)


        else:
            print("WARNING: No users had a sufficient number of relevant items")



        results_run_string = get_result_string(results_dict)




        if self.ignore_items_flag:
            recommender_object.reset_items_to_ignore()


        return (results_dict, results_run_string)




class EvaluatorNegativeItemSample(Evaluator):
    """EvaluatorNegativeItemSample"""

    EVALUATOR_NAME = "EvaluatorNegativeItemSample"

    def __init__(self, URM_test_list, URM_test_negative, cutoff_list, minRatingsPerUser=1, exclude_seen=True,
                 diversity_object = None,
                 ignore_items = None,
                 ignore_users = None):
        """

        The EvaluatorNegativeItemSample computes the recommendations by sorting the test items as well as the test_negative items
        It ensures that each item appears only once even if it is listed in both matrices

        :param URM_test_list:
        :param URM_test_negative: Items to rank together with the test items
        :param cutoff_list:
        :param minRatingsPerUser:
        :param exclude_seen:
        :param diversity_object:
        :param ignore_items:
        :param ignore_users:
        """


        super(EvaluatorNegativeItemSample, self).__init__(URM_test_list, cutoff_list,
                                                          diversity_object = diversity_object,
                                                          minRatingsPerUser=minRatingsPerUser, exclude_seen=exclude_seen,
                                                          ignore_items = ignore_items, ignore_users = ignore_users)


        self.URM_items_to_rank = sps.csr_matrix(self.URM_test.copy().astype(np.bool)) + sps.csr_matrix(URM_test_negative.copy().astype(np.bool))
        self.URM_items_to_rank.eliminate_zeros()
        self.URM_items_to_rank.data = np.ones_like(self.URM_items_to_rank.data)



    def _get_user_specific_items_to_compute(self, user_id):

        start_pos = self.URM_items_to_rank.indptr[user_id]
        end_pos = self.URM_items_to_rank.indptr[user_id+1]

        items_to_compute = self.URM_items_to_rank.indices[start_pos:end_pos]

        return items_to_compute




    def evaluateRecommender(self, recommender_object):
        """
        :param recommender_object: the trained recommender object, a BaseRecommender subclass
        :param URM_test_list: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """



        results_dict = {}

        for cutoff in self.cutoff_list:
            results_dict[cutoff] = create_empty_metrics_dict(self.n_items, self.n_users,
                                                             recommender_object.URM_train,
                                                             self.ignore_items_ID,
                                                             self.ignore_users_ID,
                                                             cutoff,
                                                             self.diversity_object)

        start_time = time.time()
        start_time_print = time.time()

        n_users_evaluated = 0

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)


        for test_user in self.usersToEvaluate:

            # Being the URM CSR, the indices are the non-zero column indexes
            relevant_items = self.get_user_relevant_items(test_user)
            relevant_items_rating = self.get_user_test_ratings(test_user)

            n_users_evaluated += 1

            items_to_compute = self._get_user_specific_items_to_compute(test_user)

            recommended_items, all_items_predicted_ratings = recommender_object.recommend(np.atleast_1d(test_user),
                                                              remove_seen_flag=self.exclude_seen,
                                                              cutoff = self.max_cutoff,
                                                              remove_top_pop_flag=False,
                                                              items_to_compute = items_to_compute,
                                                              remove_CustomItems_flag=self.ignore_items_flag,
                                                              return_scores = True
                                                             )


            assert len(recommended_items) == 1, "{}: recommended_items contained recommendations for {} users, expected was {}".format(
                self.EVALUATOR_NAME, len(recommended_items), 1)

            assert all_items_predicted_ratings.shape[0] == 1, "{}: all_items_predicted_ratings contained scores for {} users, expected was {}".format(
                self.EVALUATOR_NAME, all_items_predicted_ratings.shape[0], 1)

            assert all_items_predicted_ratings.shape[1] == self.n_items, "{}: all_items_predicted_ratings contained scores for {} items, expected was {}".format(
                self.EVALUATOR_NAME, all_items_predicted_ratings.shape[1], self.n_items)



            recommended_items = np.array(recommended_items[0])
            user_rmse = rmse(all_items_predicted_ratings[0], relevant_items, relevant_items_rating)

            recommender_object.reset_items_to_ignore()

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)



            for cutoff in self.cutoff_list:

                results_current_cutoff = results_dict[cutoff]

                is_relevant_current_cutoff = is_relevant[0:cutoff]
                recommended_items_current_cutoff = recommended_items[0:cutoff]

                results_current_cutoff[EvaluatorMetrics.ROC_AUC.value]              += roc_auc(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.PRECISION.value]            += precision(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.PRECISION_RECALL_MIN_DEN.value]   += precision_recall_min_denominator(is_relevant_current_cutoff, len(relevant_items))
                results_current_cutoff[EvaluatorMetrics.RECALL.value]               += recall(is_relevant_current_cutoff, relevant_items)
                results_current_cutoff[EvaluatorMetrics.NDCG.value]                 += ndcg(recommended_items_current_cutoff, relevant_items, relevance=self.get_user_test_ratings(test_user), at=cutoff)
                results_current_cutoff[EvaluatorMetrics.HIT_RATE.value]             += is_relevant_current_cutoff.sum()
                results_current_cutoff[EvaluatorMetrics.ARHR.value]                 += arhr(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.RMSE.value]                 += user_rmse

                results_current_cutoff[EvaluatorMetrics.MRR.value].add_recommendations(is_relevant_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.MAP.value].add_recommendations(is_relevant_current_cutoff, relevant_items)
                results_current_cutoff[EvaluatorMetrics.NOVELTY.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.AVERAGE_POPULARITY.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_GINI.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.SHANNON_ENTROPY.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.COVERAGE_ITEM.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.COVERAGE_USER.value].add_recommendations(recommended_items_current_cutoff, test_user)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST.value].add_recommendations(recommended_items_current_cutoff)
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_HERFINDAHL.value].add_recommendations(recommended_items_current_cutoff)

                if EvaluatorMetrics.DIVERSITY_SIMILARITY.value in results_current_cutoff:
                    results_current_cutoff[EvaluatorMetrics.DIVERSITY_SIMILARITY.value].add_recommendations(recommended_items_current_cutoff)



            if time.time() - start_time_print > 30 or n_users_evaluated==len(self.usersToEvaluate):
                elapsed_time = time.time()-start_time
                new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

                print("{}: Processed {} ( {:.2f}% ) in {:.2f} {}. Users per second: {:.0f}".format(
                              self.EVALUATOR_NAME,
                              n_users_evaluated,
                              100.0* float(n_users_evaluated)/len(self.usersToEvaluate),
                              new_time_value, new_time_unit,
                              float(n_users_evaluated)/elapsed_time))


                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print = time.time()


        if (n_users_evaluated > 0):

            for cutoff in self.cutoff_list:

                results_current_cutoff = results_dict[cutoff]

                for key in results_current_cutoff.keys():

                    value = results_current_cutoff[key]

                    if isinstance(value, Metrics_Object):
                        results_current_cutoff[key] = value.get_metric_value()
                    else:
                        results_current_cutoff[key] = value/n_users_evaluated

                precision_ = results_current_cutoff[EvaluatorMetrics.PRECISION.value]
                recall_ = results_current_cutoff[EvaluatorMetrics.RECALL.value]

                if precision_ + recall_ != 0:
                    # F1 micro averaged: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.8244&rep=rep1&type=pdf
                    results_current_cutoff[EvaluatorMetrics.F1.value] = 2 * (precision_ * recall_) / (precision_ + recall_)


        else:
            print("WARNING: No users had a sufficient number of relevant items")


        if self.ignore_items_flag:
            recommender_object.reset_items_to_ignore()



        results_run_string = get_result_string(results_dict)

        return (results_dict, results_run_string)