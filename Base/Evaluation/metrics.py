#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Maurizio Ferrari Dacrema, Massimo Quadrana
"""


import numpy as np
import unittest

class Metrics_Object(object):
    """
    Abstract class that should be used as superclass of all metrics requiring an object, therefore a state, to be computed
    """
    def __init__(self):
        pass

    def add_recommendations(self, recommended_items_ids):
        raise NotImplementedError()

    def get_metric_value(self):
        raise NotImplementedError()

    def merge_with_other(self, other_metric_object):
        raise NotImplementedError()



class Coverage_Item(Metrics_Object):
    """
    Item coverage represents the percentage of the overall items which were recommended
    https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff
    """

    def __init__(self, n_items, ignore_items):
        super(Coverage_Item, self).__init__()
        self.recommended_mask = np.zeros(n_items, dtype=np.bool)
        self.n_ignore_items = len(ignore_items)

    def add_recommendations(self, recommended_items_ids):
        if len(recommended_items_ids) > 0:
            self.recommended_mask[recommended_items_ids] = True

    def get_metric_value(self):
        return self.recommended_mask.sum()/(len(self.recommended_mask)-self.n_ignore_items)


    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Coverage_Item, "Coverage_Item: attempting to merge with a metric object of different type"

        self.recommended_mask = np.logical_or(self.recommended_mask, other_metric_object.recommended_mask)




class Coverage_User(Metrics_Object):
    """
    User coverage represents the percentage of the overall users for which we can make recommendations.
    If there is at least one recommendation the user is considered as covered
    https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff
    """

    def __init__(self, n_users, ignore_users):
        super(Coverage_User, self).__init__()
        self.users_mask = np.zeros(n_users, dtype=np.bool)
        self.n_ignore_users = len(ignore_users)

    def add_recommendations(self, recommended_items_ids, user_id):
        self.users_mask[user_id] = len(recommended_items_ids)>0

    def get_metric_value(self):
        return self.users_mask.sum()/(len(self.users_mask)-self.n_ignore_users)

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Coverage_User, "Coverage_User: attempting to merge with a metric object of different type"

        self.users_mask = np.logical_or(self.users_mask, other_metric_object.users_mask)




class MAP(Metrics_Object):
    """
    Mean Average Precision, defined as the mean of the AveragePrecision over all users

    """

    def __init__(self):
        super(MAP, self).__init__()
        self.cumulative_AP = 0.0
        self.n_users = 0

    def add_recommendations(self, is_relevant, pos_items):
        self.cumulative_AP += average_precision(is_relevant, pos_items)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_AP/self.n_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is MAP, "MAP: attempting to merge with a metric object of different type"

        self.cumulative_AP += other_metric_object.cumulative_AP
        self.n_users += other_metric_object.n_users





class MRR(Metrics_Object):
    """
    Mean Reciprocal Rank, defined as the mean of the Reciprocal Rank over all users

    """

    def __init__(self):
        super(MRR, self).__init__()
        self.cumulative_RR = 0.0
        self.n_users = 0

    def add_recommendations(self, is_relevant):
        self.cumulative_RR += rr(is_relevant)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_RR/self.n_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is MAP, "MRR: attempting to merge with a metric object of different type"

        self.cumulative_RR += other_metric_object.cumulative_RR
        self.n_users += other_metric_object.n_users





class Gini_Diversity(Metrics_Object):
    """
    Gini diversity index, computed from the Gini Index but with inverted range, such that high values mean higher diversity
    This implementation ignores zero-occurrence items

    # From https://github.com/oliviaguest/gini
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    #
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.8174&rep=rep1&type=pdf
    """

    def __init__(self, n_items, ignore_items):
        super(Gini_Diversity, self).__init__()
        self.recommended_counter = np.zeros(n_items, dtype=np.float)
        self.ignore_items = ignore_items.astype(np.int).copy()

    def add_recommendations(self, recommended_items_ids):
        if len(recommended_items_ids) > 0:
            self.recommended_counter[recommended_items_ids] += 1

    def get_metric_value(self):

        recommended_counter = self.recommended_counter.copy()

        recommended_counter_mask = np.ones_like(recommended_counter, dtype = np.bool)
        recommended_counter_mask[self.ignore_items] = False
        recommended_counter_mask[recommended_counter == 0] = False

        recommended_counter = recommended_counter[recommended_counter_mask]

        n_items = len(recommended_counter)

        recommended_counter_sorted = np.sort(recommended_counter)       # values must be sorted
        index = np.arange(1, n_items+1)                                 # index per array element

        #gini_index = (np.sum((2 * index - n_items  - 1) * recommended_counter_sorted)) / (n_items * np.sum(recommended_counter_sorted))
        gini_diversity = 2*np.sum((n_items + 1 - index)/(n_items+1) * recommended_counter_sorted/np.sum(recommended_counter_sorted))

        return gini_diversity

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Gini_Diversity, "Gini_Diversity: attempting to merge with a metric object of different type"

        self.recommended_counter += other_metric_object.recommended_counter




class Diversity_Herfindahl(Metrics_Object):
    """
    The Herfindahl index is also known as Concentration index, it is used in economy to determine whether the market quotas
    are such that an excessive concentration exists. It is here used as a diversity index, if high means high diversity.

    It is known to have a small value range in recommender systems, between 0.9 and 1.0

    The Herfindahl index is a function of the square of the probability an item has been recommended to any user, hence
    The Herfindahl index is equivalent to MeanInterList diversity as they measure the same quantity.

    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.8174&rep=rep1&type=pdf
    """

    def __init__(self, n_items, ignore_items):
        super(Diversity_Herfindahl, self).__init__()
        self.recommended_counter = np.zeros(n_items, dtype=np.float)
        self.ignore_items = ignore_items.astype(np.int).copy()

    def add_recommendations(self, recommended_items_ids):
        if len(recommended_items_ids) > 0:
            self.recommended_counter[recommended_items_ids] += 1

    def get_metric_value(self):

        recommended_counter = self.recommended_counter.copy()

        recommended_counter_mask = np.ones_like(recommended_counter, dtype = np.bool)
        recommended_counter_mask[self.ignore_items] = False

        recommended_counter = recommended_counter[recommended_counter_mask]

        if recommended_counter.sum() != 0:
            herfindahl_index = 1 - np.sum((recommended_counter / recommended_counter.sum()) ** 2)
        else:
            herfindahl_index = np.nan

        return herfindahl_index

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Diversity_Herfindahl, "Diversity_Herfindahl: attempting to merge with a metric object of different type"

        self.recommended_counter += other_metric_object.recommended_counter





class Shannon_Entropy(Metrics_Object):
    """
    Shannon Entropy is a well known metric to measure the amount of information of a certain string of data.
    Here is applied to the global number of times an item has been recommended.

    It has a lower bound and can reach values over 12.0 for random recommenders.
    A high entropy means that the distribution is random uniform across all users.

    Note that while a random uniform distribution
    (hence all items with SIMILAR number of occurrences)
    will be highly diverse and have high entropy, a perfectly uniform distribution
    (hence all items with EXACTLY IDENTICAL number of occurrences)
    will have 0.0 entropy while being the most diverse possible.

    """

    def __init__(self, n_items, ignore_items):
        super(Shannon_Entropy, self).__init__()
        self.recommended_counter = np.zeros(n_items, dtype=np.float)
        self.ignore_items = ignore_items.astype(np.int).copy()

    def add_recommendations(self, recommended_items_ids):
        if len(recommended_items_ids) > 0:
            self.recommended_counter[recommended_items_ids] += 1

    def get_metric_value(self):

        assert np.all(self.recommended_counter >= 0.0), "Shannon_Entropy: self.recommended_counter contains negative counts"

        recommended_counter = self.recommended_counter.copy()

        # Ignore from the computation both ignored items and items with zero occurrence.
        # Zero occurrence items will have zero probability and will not change the result, butt will generate nans if used in the log
        recommended_counter_mask = np.ones_like(recommended_counter, dtype = np.bool)
        recommended_counter_mask[self.ignore_items] = False
        recommended_counter_mask[recommended_counter == 0] = False

        recommended_counter = recommended_counter[recommended_counter_mask]

        n_recommendations = recommended_counter.sum()

        recommended_probability = recommended_counter/n_recommendations

        shannon_entropy = -np.sum(recommended_probability * np.log2(recommended_probability))

        return shannon_entropy

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Gini_Diversity, "Shannon_Entropy: attempting to merge with a metric object of different type"

        assert np.all(self.recommended_counter >= 0.0), "Shannon_Entropy: self.recommended_counter contains negative counts"
        assert np.all(other_metric_object.recommended_counter >= 0.0), "Shannon_Entropy: other.recommended_counter contains negative counts"

        self.recommended_counter += other_metric_object.recommended_counter





import scipy.sparse as sps



class Novelty(Metrics_Object):
    """
    Novelty measures how "novel" a recommendation is in terms of how popular the item was in the train set.

    Due to this definition, the novelty of a cold item (i.e. with no interactions in the train set) is not defined,
    in this implementation cold items are ignored and their contribution to the novelty is 0.

    A recommender with high novelty will be able to recommend also long queue (i.e. unpopular) items.

    Mean self-information  (Zhou 2010)
    """

    def __init__(self, URM_train):
        super(Novelty, self).__init__()

        URM_train = sps.csc_matrix(URM_train)
        URM_train.eliminate_zeros()
        self.item_popularity = np.ediff1d(URM_train.indptr)

        self.novelty = 0.0
        self.n_evaluated_users = 0
        self.n_items = len(self.item_popularity)
        self.n_interactions = self.item_popularity.sum()


    def add_recommendations(self, recommended_items_ids):

        self.n_evaluated_users += 1

        if len(recommended_items_ids)>0:
            recommended_items_popularity = self.item_popularity[recommended_items_ids]

            probability = recommended_items_popularity/self.n_interactions
            probability = probability[probability!=0]

            self.novelty += np.sum(-np.log2(probability)/self.n_items)


    def get_metric_value(self):

        if self.n_evaluated_users == 0:
            return 0.0

        return self.novelty/self.n_evaluated_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Novelty, "Novelty: attempting to merge with a metric object of different type"

        self.novelty = self.novelty + other_metric_object.novelty
        self.n_evaluated_users = self.n_evaluated_users + other_metric_object.n_evaluated_users







class AveragePopularity(Metrics_Object):
    """
    Average popularity the recommended items have in the train data.
    The popularity is normalized by setting as 1 the item with the highest popularity in the train data
    """

    def __init__(self, URM_train):
        super(AveragePopularity, self).__init__()

        URM_train = sps.csc_matrix(URM_train)
        URM_train.eliminate_zeros()
        item_popularity = np.ediff1d(URM_train.indptr)


        self.cumulative_popularity = 0.0
        self.n_evaluated_users = 0
        self.n_items = URM_train.shape[0]
        self.n_interactions = item_popularity.sum()

        self.item_popularity_normalized = item_popularity/item_popularity.max()


    def add_recommendations(self, recommended_items_ids):

        self.n_evaluated_users += 1

        if len(recommended_items_ids)>0:
            recommended_items_popularity = self.item_popularity_normalized[recommended_items_ids]

            self.cumulative_popularity += np.sum(recommended_items_popularity)/len(recommended_items_ids)


    def get_metric_value(self):

        if self.n_evaluated_users == 0:
            return 0.0

        return self.cumulative_popularity/self.n_evaluated_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Novelty, "AveragePopularity: attempting to merge with a metric object of different type"

        self.cumulative_popularity = self.cumulative_popularity + other_metric_object.cumulative_popularity
        self.n_evaluated_users = self.n_evaluated_users + other_metric_object.n_evaluated_users






class Diversity_similarity(Metrics_Object):
    """
    Intra list diversity computes the diversity of items appearing in the recommendations received by each single user, by using an item_diversity_matrix.

    It can be used, for example, to compute the diversity in terms of features for a collaborative recommender.

    A content-based recommender will have low IntraList diversity if that is computed on the same features the recommender uses.
    A TopPopular recommender may exhibit high IntraList diversity.

    """

    def __init__(self, item_diversity_matrix):
        super(Diversity_similarity, self).__init__()

        assert np.all(item_diversity_matrix >= 0.0) and np.all(item_diversity_matrix <= 1.0), \
            "item_diversity_matrix contains value greated than 1.0 or lower than 0.0"

        self.item_diversity_matrix = item_diversity_matrix

        self.n_evaluated_users = 0
        self.diversity = 0.0


    def add_recommendations(self, recommended_items_ids):

        current_recommended_items_diversity = 0.0

        for item_index in range(len(recommended_items_ids)-1):

            item_id = recommended_items_ids[item_index]

            item_other_diversity = self.item_diversity_matrix[item_id, recommended_items_ids]
            item_other_diversity[item_index] = 0.0

            current_recommended_items_diversity += np.sum(item_other_diversity)


        self.diversity += current_recommended_items_diversity/(len(recommended_items_ids)*(len(recommended_items_ids)-1))

        self.n_evaluated_users += 1


    def get_metric_value(self):

        if self.n_evaluated_users == 0:
            return 0.0

        return self.diversity/self.n_evaluated_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Diversity_similarity, "Diversity: attempting to merge with a metric object of different type"

        self.diversity = self.diversity + other_metric_object.diversity
        self.n_evaluated_users = self.n_evaluated_users + other_metric_object.n_evaluated_users




class Diversity_MeanInterList(Metrics_Object):
    """
    MeanInterList diversity measures the uniqueness of different users' recommendation lists.

    It can be used to measure how "diversified" are the recommendations different users receive.

    While the original proposal called this metric "Personalization", we do not use this name since the highest MeanInterList diversity
    is exhibited by a non personalized Random recommender.

    It can be demonstrated that this metric does not require to compute the common items all possible couples of users have in common
    but rather it is only sensitive to the total amount of time each item has been recommended.

    MeanInterList diversity is a function of the square of the probability an item has been recommended to any user, hence
    MeanInterList diversity is equivalent to the Herfindahl index as they measure the same quantity.

    A TopPopular recommender that does not remove seen items will have 0.0 MeanInterList diversity.


    pag. 3, http://www.pnas.org/content/pnas/107/10/4511.full.pdf

    @article{zhou2010solving,
      title={Solving the apparent diversity-accuracy dilemma of recommender systems},
      author={Zhou, Tao and Kuscsik, Zolt{\'a}n and Liu, Jian-Guo and Medo, Mat{\'u}{\v{s}} and Wakeling, Joseph Rushton and Zhang, Yi-Cheng},
      journal={Proceedings of the National Academy of Sciences},
      volume={107},
      number={10},
      pages={4511--4515},
      year={2010},
      publisher={National Acad Sciences}
    }

    # The formula is diversity_cumulative += 1 - common_recommendations(user1, user2)/cutoff
    # for each couple of users, except the diagonal. It is VERY computationally expensive
    # We can move the 1 and cutoff outside of the summation. Remember to exclude the diagonal
    # co_counts = URM_predicted.dot(URM_predicted.T)
    # co_counts[np.arange(0, n_user, dtype=np.int):np.arange(0, n_user, dtype=np.int)] = 0
    # diversity = (n_user**2 - n_user) - co_counts.sum()/self.cutoff

    # If we represent the summation of co_counts separating it for each item, we will have:
    # co_counts.sum() = co_counts_item1.sum()  + co_counts_item2.sum() ...
    # If we know how many times an item has been recommended, co_counts_item1.sum() can be computed as how many couples of
    # users have item1 in common. If item1 has been recommended n times, the number of couples is n*(n-1)
    # Therefore we can compute co_counts.sum() value as:
    # np.sum(np.multiply(item-occurrence, item-occurrence-1))

    # The naive implementation URM_predicted.dot(URM_predicted.T) might require an hour of computation
    # The last implementation has a negligible computational time even for very big datasets

    """

    def __init__(self, n_items, cutoff):
        super(Diversity_MeanInterList, self).__init__()

        self.recommended_counter = np.zeros(n_items, dtype=np.float)

        self.n_evaluated_users = 0
        self.n_items = n_items
        self.diversity = 0.0
        self.cutoff = cutoff


    def add_recommendations(self, recommended_items_ids):

        assert len(recommended_items_ids) <= self.cutoff, "Diversity_MeanInterList: recommended list is contains more elements than cutoff"

        self.n_evaluated_users += 1

        if len(recommended_items_ids) > 0:
            self.recommended_counter[recommended_items_ids] += 1




    def get_metric_value(self):

        # Requires to compute the number of common elements for all couples of users
        if self.n_evaluated_users == 0:
            return 1.0

        cooccurrences_cumulative = np.sum(self.recommended_counter**2) - self.n_evaluated_users*self.cutoff

        # All user combinations except diagonal
        all_user_couples_count = self.n_evaluated_users**2 - self.n_evaluated_users

        diversity_cumulative = all_user_couples_count - cooccurrences_cumulative/self.cutoff

        self.diversity = diversity_cumulative/all_user_couples_count

        return self.diversity


    def get_theoretical_max(self):

        global_co_occurrence_count = (self.n_evaluated_users*self.cutoff)**2/self.n_items - self.n_evaluated_users*self.cutoff

        mild = 1 - 1/(self.n_evaluated_users**2 - self.n_evaluated_users)*(global_co_occurrence_count/self.cutoff)

        return mild

    def merge_with_other(self, other_metric_object):

        assert other_metric_object is Diversity_MeanInterList, "Diversity_MeanInterList: attempting to merge with a metric object of different type"

        assert np.all(self.recommended_counter >= 0.0), "Diversity_MeanInterList: self.recommended_counter contains negative counts"
        assert np.all(other_metric_object.recommended_counter >= 0.0), "Diversity_MeanInterList: other.recommended_counter contains negative counts"

        self.recommended_counter += other_metric_object.recommended_counter
        self.n_evaluated_users += other_metric_object.n_evaluated_users





def roc_auc(is_relevant):

    ranks = np.arange(len(is_relevant))
    pos_ranks = ranks[is_relevant]
    neg_ranks = ranks[~is_relevant]
    auc_score = 0.0

    if len(neg_ranks) == 0:
        return 1.0

    if len(pos_ranks) > 0:
        for pos_pred in pos_ranks:
            auc_score += np.sum(pos_pred < neg_ranks, dtype=np.float32)
        auc_score /= (pos_ranks.shape[0] * neg_ranks.shape[0])

    assert 0 <= auc_score <= 1, auc_score
    return auc_score



def arhr(is_relevant):
    # average reciprocal hit-rank (ARHR) of all relevant items
    # As opposed to MRR, ARHR takes into account all relevant items and not just the first
    # pag 17
    # http://glaros.dtc.umn.edu/gkhome/fetch/papers/itemrsTOIS04.pdf
    # https://emunix.emich.edu/~sverdlik/COSC562/ItemBasedTopTen.pdf

    p_reciprocal = 1/np.arange(1,len(is_relevant)+1, 1.0, dtype=np.float64)
    arhr_score = is_relevant.dot(p_reciprocal)

    #assert 0 <= arhr_score <= p_reciprocal.sum(), "arhr_score {} should be between 0 and {}".format(arhr_score, p_reciprocal.sum())
    assert not np.isnan(arhr_score), "ARHR is NaN"
    return arhr_score



def precision(is_relevant):

    if len(is_relevant) == 0:
        precision_score = 0.0
    else:
        precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    assert 0 <= precision_score <= 1, precision_score
    return precision_score


def precision_recall_min_denominator(is_relevant, n_test_items):

    if len(is_relevant) == 0:
        precision_score = 0.0
    else:
        precision_score = np.sum(is_relevant, dtype=np.float32) / min(n_test_items, len(is_relevant))

    assert 0 <= precision_score <= 1, precision_score
    return precision_score


def rmse(all_items_predicted_ratings, relevant_items, relevant_items_rating):

    # Important, some items will have -np.inf score and are treated as if they did not exist

    # RMSE with test items
    relevant_items_error = (all_items_predicted_ratings[relevant_items]-relevant_items_rating)**2

    finite_prediction_mask = np.isfinite(relevant_items_error)

    if finite_prediction_mask.sum() == 0:
        rmse = np.nan

    else:
        relevant_items_error = relevant_items_error[finite_prediction_mask]

        squared_error = np.sum(relevant_items_error)

        # # Second the RMSE against all non-test items assumed having true rating 0
        # # In order to avoid the need of explicitly indexing all non-relevant items, use a difference
        # squared_error += np.sum(all_items_predicted_ratings[np.isfinite(all_items_predicted_ratings)]**2) - \
        #                  np.sum(all_items_predicted_ratings[relevant_items][np.isfinite(all_items_predicted_ratings[relevant_items])]**2)

        mean_squared_error = squared_error/finite_prediction_mask.sum()
        rmse = np.sqrt(mean_squared_error)

    return rmse


def recall(is_relevant, pos_items):

    recall_score = np.sum(is_relevant, dtype=np.float32) / pos_items.shape[0]

    assert 0 <= recall_score <= 1, recall_score
    return recall_score


def rr(is_relevant):
    # reciprocal rank of the FIRST relevant item in the ranked list (0 if none)

    ranks = np.arange(1, len(is_relevant) + 1)[is_relevant]

    if len(ranks) > 0:
        return 1. / ranks[0]
    else:
        return 0.0


def average_precision(is_relevant, pos_items):

    if len(is_relevant) == 0:
        a_p = 0.0
    else:
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
        a_p = np.sum(p_at_k) / np.min([pos_items.shape[0], is_relevant.shape[0]])

    assert 0 <= a_p <= 1, a_p
    return a_p


def ndcg(ranked_list, pos_items, relevance=None, at=None):

    if relevance is None:
        relevance = np.ones_like(pos_items)
    assert len(relevance) == pos_items.shape[0]

    # Create a dictionary associating item_id to its relevance
    # it2rel[item] -> relevance[item]
    it2rel = {it: r for it, r in zip(pos_items, relevance)}

    # Creates array of length "at" with the relevance associated to the item in that position
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float32)

    # IDCG has all relevances to 1, up to the number of items in the test set
    ideal_dcg = dcg(np.sort(relevance)[::-1])

    # DCG uses the relevance of the recommended items
    rank_dcg = dcg(rank_scores)

    if rank_dcg == 0.0:
        return 0.0

    ndcg_ = rank_dcg / ideal_dcg
    # assert 0 <= ndcg_ <= 1, (rank_dcg, ideal_dcg, ndcg_)
    return ndcg_


def dcg(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                  dtype=np.float32)


metrics = ['AUC', 'Precision' 'Recall', 'MAP', 'NDCG']


def pp_metrics(metric_names, metric_values, metric_at):
    """
    Pretty-prints metric values
    :param metrics_arr:
    :return:
    """
    assert len(metric_names) == len(metric_values)
    if isinstance(metric_at, int):
        metric_at = [metric_at] * len(metric_values)
    return ' '.join(['{}: {:.4f}'.format(mname, mvalue) if mcutoff is None or mcutoff == 0 else
                     '{}@{}: {:.4f}'.format(mname, mcutoff, mvalue)
                     for mname, mcutoff, mvalue in zip(metric_names, metric_at, metric_values)])


class TestAUC(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4])
        ranked_list = np.asarray([1, 2, 3, 4, 5])
        self.assertTrue(np.allclose(roc_auc(ranked_list, pos_items),
                                    (2. / 3 + 1. / 3) / 2))


class TestRecall(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        self.assertTrue(np.allclose(recall(ranked_list_1, pos_items), 3. / 4))
        self.assertTrue(np.allclose(recall(ranked_list_2, pos_items), 1.0))
        self.assertTrue(np.allclose(recall(ranked_list_3, pos_items), 0.0))

        thresholds = [1, 2, 3, 4, 5]
        values = [0.0, 1. / 4, 1. / 4, 2. / 4, 3. / 4]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(recall(ranked_list_1, pos_items, at=at)), val))


class TestPrecision(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        self.assertTrue(np.allclose(precision(ranked_list_1, pos_items), 3. / 5))
        self.assertTrue(np.allclose(precision(ranked_list_2, pos_items), 4. / 5))
        self.assertTrue(np.allclose(precision(ranked_list_3, pos_items), 0.0))

        thresholds = [1, 2, 3, 4, 5]
        values = [0.0, 1. / 2, 1. / 3, 2. / 4, 3. / 5]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(precision(ranked_list_1, pos_items, at=at)), val))


class TestRR(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        self.assertTrue(np.allclose(rr(ranked_list_1, pos_items), 1. / 2))
        self.assertTrue(np.allclose(rr(ranked_list_2, pos_items), 1.))
        self.assertTrue(np.allclose(rr(ranked_list_3, pos_items), 0.0))

        thresholds = [1, 2, 3, 4, 5]
        values = [0.0, 1. / 2, 1. / 2, 1. / 2, 1. / 2]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(rr(ranked_list_1, pos_items, at=at)), val))


class TestMAP(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        ranked_list_4 = np.asarray([11, 12, 13, 14, 15, 16, 2, 4, 5, 10])
        ranked_list_5 = np.asarray([2, 11, 12, 13, 14, 15, 4, 5, 10, 16])
        self.assertTrue(np.allclose(map(ranked_list_1, pos_items), (1. / 2 + 2. / 4 + 3. / 5) / 4))
        self.assertTrue(np.allclose(map(ranked_list_2, pos_items), 1.0))
        self.assertTrue(np.allclose(map(ranked_list_3, pos_items), 0.0))
        self.assertTrue(np.allclose(map(ranked_list_4, pos_items), (1. / 7 + 2. / 8 + 3. / 9 + 4. / 10) / 4))
        self.assertTrue(np.allclose(map(ranked_list_5, pos_items), (1. + 2. / 7 + 3. / 8 + 4. / 9) / 4))

        thresholds = [1, 2, 3, 4, 5]
        values = [
            0.0,
            1. / 2 / 2,
            1. / 2 / 3,
            (1. / 2 + 2. / 4) / 4,
            (1. / 2 + 2. / 4 + 3. / 5) / 4
        ]
        for at, val in zip(thresholds, values):
            self.assertTrue(np.allclose(np.asarray(map(ranked_list_1, pos_items, at)), val))


class TestNDCG(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        pos_relevances = np.asarray([5, 4, 3, 2])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])  # rel = 0, 5, 0, 4, 3
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])  # rel = 2, 3, 5, 4, 0
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])  # rel = 0, 0, 0, 0, 0
        idcg = ((2 ** 5 - 1) / np.log(2) +
                (2 ** 4 - 1) / np.log(3) +
                (2 ** 3 - 1) / np.log(4) +
                (2 ** 2 - 1) / np.log(5))
        self.assertTrue(np.allclose(dcg(np.sort(pos_relevances)[::-1]), idcg))
        self.assertTrue(np.allclose(ndcg(ranked_list_1, pos_items, pos_relevances),
                                    ((2 ** 5 - 1) / np.log(3) +
                                     (2 ** 4 - 1) / np.log(5) +
                                     (2 ** 3 - 1) / np.log(6)) / idcg))
        self.assertTrue(np.allclose(ndcg(ranked_list_2, pos_items, pos_relevances),
                                    ((2 ** 2 - 1) / np.log(2) +
                                     (2 ** 3 - 1) / np.log(3) +
                                     (2 ** 5 - 1) / np.log(4) +
                                     (2 ** 4 - 1) / np.log(5)) / idcg))
        self.assertTrue(np.allclose(ndcg(ranked_list_3, pos_items, pos_relevances), 0.0))


if __name__ == '__main__':
    unittest.main()
