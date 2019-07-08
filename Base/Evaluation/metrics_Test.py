#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/09/17

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import unittest



class MyTestCase(unittest.TestCase):

    def test_Gini_Index(self):

        from Base.Evaluation.metrics import Gini_Diversity

        n_items = 1000

        gini_index = Gini_Diversity(n_items, ignore_items=np.array([]))

        gini_index.recommended_counter = np.ones(n_items)
        assert np.isclose(1.0, gini_index.get_metric_value(), atol=1e-2), "Gini_Index metric incorrect"

        gini_index.recommended_counter = np.ones(n_items)*1e-12
        gini_index.recommended_counter[0] = 1.0
        assert np.isclose(0.0, gini_index.get_metric_value(), atol=1e-2), "Gini_Index metric incorrect"

        # gini_index.recommended_counter = np.random.uniform(0, 1, n_items)
        # assert  np.isclose(0.3, gini_index.get_metric_value(), atol=1e-1), "Gini_Index metric incorrect"



    def test_Shannon_Entropy(self):

        from Base.Evaluation.metrics import Shannon_Entropy

        n_items = 1000

        shannon_entropy = Shannon_Entropy(n_items, ignore_items=np.array([]))

        shannon_entropy.recommended_counter = np.ones(n_items)
        assert np.isclose(9.96, shannon_entropy.get_metric_value(), atol=1e-2), "metric incorrect"

        shannon_entropy.recommended_counter = np.zeros(n_items)
        shannon_entropy.recommended_counter[0] = 1.0
        assert np.isclose(0.0, shannon_entropy.get_metric_value(), atol=1e-3), "metric incorrect"

        shannon_entropy.recommended_counter = np.random.uniform(0, 100, n_items).astype(np.int)
        assert  np.isclose(9.6, shannon_entropy.get_metric_value(), atol=1e-1), "metric incorrect"

        # n_items = 10000
        #
        # shannon_entropy.recommended_counter = np.random.normal(0, 50, n_items).astype(np.int)
        # shannon_entropy.recommended_counter += abs(min(shannon_entropy.recommended_counter))
        # assert  np.isclose(9.8, shannon_entropy.get_metric_value(), atol=1e-1), "metric incorrect"




    def test_Diversity_list_all_equals(self):

        from Base.Evaluation.metrics import Diversity_MeanInterList
        import scipy.sparse as sps

        n_items = 3
        n_users = 10
        cutoff = min(5, n_items)

        # create recommendation list
        URM_predicted_row = []
        URM_predicted_col = []

        diversity_list = Diversity_MeanInterList(n_items, cutoff)
        item_id_list = np.arange(0, n_items, dtype=np.int)

        for n_user in range(n_users):

            np.random.shuffle(item_id_list)
            recommended = item_id_list[:cutoff]
            URM_predicted_row.extend([n_user]*cutoff)
            URM_predicted_col.extend(recommended)

            diversity_list.add_recommendations(recommended)


        object_diversity = diversity_list.get_metric_value()

        URM_predicted_data = np.ones_like(URM_predicted_row)

        URM_predicted_sparse = sps.csr_matrix((URM_predicted_data, (URM_predicted_row, URM_predicted_col)), dtype=np.int)

        co_counts = URM_predicted_sparse.dot(URM_predicted_sparse.T).toarray()
        np.fill_diagonal(co_counts, 0)

        all_user_couples_count = n_users**2 - n_users

        diversity_cumulative = 1 - co_counts/cutoff
        np.fill_diagonal(diversity_cumulative, 0)

        diversity_cooccurrence = diversity_cumulative.sum()/all_user_couples_count

        assert  np.isclose(diversity_cooccurrence, object_diversity, atol=1e-4), "metric incorrect"






    def test_Diversity_list(self):

        from Base.Evaluation.metrics import Diversity_MeanInterList
        import scipy.sparse as sps

        n_items = 500
        n_users = 1000
        cutoff = 10

        # create recommendation list
        URM_predicted_row = []
        URM_predicted_col = []

        diversity_list = Diversity_MeanInterList(n_items, cutoff)
        item_id_list = np.arange(0, n_items, dtype=np.int)

        for n_user in range(n_users):

            np.random.shuffle(item_id_list)
            recommended = item_id_list[:cutoff]
            URM_predicted_row.extend([n_user]*cutoff)
            URM_predicted_col.extend(recommended)

            diversity_list.add_recommendations(recommended)


        object_diversity = diversity_list.get_metric_value()

        URM_predicted_data = np.ones_like(URM_predicted_row)

        URM_predicted_sparse = sps.csr_matrix((URM_predicted_data, (URM_predicted_row, URM_predicted_col)), dtype=np.int)

        co_counts = URM_predicted_sparse.dot(URM_predicted_sparse.T).toarray()
        np.fill_diagonal(co_counts, 0)

        all_user_couples_count = n_users**2 - n_users

        diversity_cumulative = 1 - co_counts/cutoff
        np.fill_diagonal(diversity_cumulative, 0)

        diversity_cooccurrence = diversity_cumulative.sum()/all_user_couples_count

        assert  np.isclose(diversity_cooccurrence, object_diversity, atol=1e-4), "metric incorrect"


    def test_AUC(self):

        from Base.Evaluation.metrics import roc_auc

        pos_items = np.asarray([2, 4])
        ranked_list = np.asarray([1, 2, 3, 4, 5])

        is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)

        self.assertTrue(np.allclose(roc_auc(is_relevant),
                                    (2. / 3 + 1. / 3) / 2))



    def test_Recall(self):

        from Base.Evaluation.metrics import recall

        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])

        is_relevant = np.in1d(ranked_list_1, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(recall(is_relevant, pos_items), 3. / 4))

        is_relevant = np.in1d(ranked_list_2, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(recall(is_relevant, pos_items), 1.0))

        is_relevant = np.in1d(ranked_list_3, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(recall(is_relevant, pos_items), 0.0))

        # thresholds = [1, 2, 3, 4, 5]
        # values = [0.0, 1. / 4, 1. / 4, 2. / 4, 3. / 4]
        # for at, val in zip(thresholds, values):
        #     self.assertTrue(np.allclose(np.asarray(recall(ranked_list_1, pos_items, at=at)), val))



    def test_Precision(self):

        from Base.Evaluation.metrics import precision

        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])

        is_relevant = np.in1d(ranked_list_1, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(precision(is_relevant), 3. / 5))

        is_relevant = np.in1d(ranked_list_2, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(precision(is_relevant), 4. / 5))

        is_relevant = np.in1d(ranked_list_3, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(precision(is_relevant), 0.0))

        # thresholds = [1, 2, 3, 4, 5]
        # values = [0.0, 1. / 2, 1. / 3, 2. / 4, 3. / 5]
        # for at, val in zip(thresholds, values):
        #     self.assertTrue(np.allclose(np.asarray(precision(ranked_list_1, pos_items, at=at)), val))



    def test_RR(self):

        from Base.Evaluation.metrics import rr

        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])

        is_relevant = np.in1d(ranked_list_1, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(rr(is_relevant), 1. / 2))

        is_relevant = np.in1d(ranked_list_2, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(rr(is_relevant), 1.))

        is_relevant = np.in1d(ranked_list_3, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(rr(is_relevant), 0.0))

        # thresholds = [1, 2, 3, 4, 5]
        # values = [0.0, 1. / 2, 1. / 2, 1. / 2, 1. / 2]
        # for at, val in zip(thresholds, values):
        #     self.assertTrue(np.allclose(np.asarray(rr(ranked_list_1, pos_items, at=at)), val))



    def test_MAP(self):

        from Base.Evaluation.metrics import map

        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        ranked_list_4 = np.asarray([11, 12, 13, 14, 15, 16, 2, 4, 5, 10])
        ranked_list_5 = np.asarray([2, 11, 12, 13, 14, 15, 4, 5, 10, 16])

        is_relevant = np.in1d(ranked_list_1, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(map(is_relevant, pos_items), (1. / 2 + 2. / 4 + 3. / 5) / 4))

        is_relevant = np.in1d(ranked_list_2, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(map(is_relevant, pos_items), 1.0))

        is_relevant = np.in1d(ranked_list_3, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(map(is_relevant, pos_items), 0.0))

        is_relevant = np.in1d(ranked_list_4, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(map(is_relevant, pos_items), (1. / 7 + 2. / 8 + 3. / 9 + 4. / 10) / 4))

        is_relevant = np.in1d(ranked_list_5, pos_items, assume_unique=True)
        self.assertTrue(np.allclose(map(is_relevant, pos_items), (1. + 2. / 7 + 3. / 8 + 4. / 9) / 4))

        # thresholds = [1, 2, 3, 4, 5]
        # values = [
        #     0.0,
        #     1. / 2 / 2,
        #     1. / 2 / 3,
        #     (1. / 2 + 2. / 4) / 4,
        #     (1. / 2 + 2. / 4 + 3. / 5) / 4
        # ]
        # for at, val in zip(thresholds, values):
        #     self.assertTrue(np.allclose(np.asarray(map(ranked_list_1, pos_items, at)), val))



    def test_NDCG(self):

        from Base.Evaluation.metrics import dcg, ndcg

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

