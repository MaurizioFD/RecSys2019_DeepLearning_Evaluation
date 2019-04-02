#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21/06/2017

@author: Anonymous authors
"""
import pickle
from unittest import TestCase

from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


class TestRecommender(TestCase):

    def loadFiles(self):

        base_path = "data/"

        URM_train_file = open(base_path + "partial_urm_1_validation", 'rb')
        self.URM_train = pickle.load(URM_train_file)
        URM_train_file.close()

        URM_test_file = open(base_path + "partial_urm_2_validation", 'rb')
        self.URM_test = pickle.load(URM_test_file)
        URM_test_file.close()

        ICM_file = open(base_path + "icm_matrix.dat", 'rb')
        self.ICM = pickle.load(ICM_file).T
        ICM_file.close()

    def test_recommendBlock(self):

        self.loadFiles()

        recommender = ItemKNNCBFRecommender(k=50, shrink=2, similarity='cosine')

        recommender.fit(self.ICM, self.URM_train)

        numUsers = self.URM_train.shape[0]

        for userID in range(numUsers):

            self.assertEquals(recommender.recommend(userID), recommender.recommendBatch(userID, userID))


