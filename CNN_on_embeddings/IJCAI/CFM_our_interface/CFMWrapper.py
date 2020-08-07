#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/10/19

@author: Federico Parroni
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm
import numpy as np

from Base.DataIO import DataIO
from Base.BaseRecommender import BaseRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.BaseTempFolder import BaseTempFolder
from CNN_on_embeddings.IJCAI.CFM_our_interface.CFM import CFM



class CFM_wrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "CFM_Wrapper"
    __AVAILABLE_MAP_MODES = ["all_map", "main_diagonal", "off_diagonal"]

    def __init__(self, URM_train, dataset):
        super().__init__(URM_train)

        self.data = dataset
        self.user_field_M = self.data.user_field_M
        self.item_field_M = self.data.item_field_M
        self._item_indices = np.arange(0, self.n_items, dtype=np.int32)

        self.cfm = None
    
    def fit(self, pretrain_flag, pretrained_FM_folder_path, hidden_factor, num_field, epochs, batch_size,
            learning_rate, lamda_bilinear, keep, optimizer_type, batch_norm, verbose,
            regs, map_mode, attention_size, attentive_pooling, net_channel,
            permutation=None,
            random_seed=None,
            temp_file_folder = None, **earlystopping_kwargs):

        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)
        
        assert map_mode in self.__AVAILABLE_MAP_MODES, 'Invalid map mode!'

        self.cfm = CFM(
            data=self.data,
            user_field_M=self.user_field_M,
            item_field_M=self.item_field_M,
            pretrain_flag=pretrain_flag,
            pretrained_FM_folder_path=pretrained_FM_folder_path,
            hidden_factor=hidden_factor,
            num_field=num_field,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lamda_bilinear=lamda_bilinear,
            keep=keep,
            optimizer_type=optimizer_type,
            batch_norm=batch_norm,
            verbose=verbose,
            regs=regs,
            attention_size=attention_size,
            attentive_pooling=attentive_pooling,
            net_channel=net_channel,
            map_mode=map_mode,
            permutation=permutation,
            random_seed=random_seed
        )

        self._update_best_model()

        self._train_with_early_stopping(epochs, algorithm_name=self.RECOMMENDER_NAME, **earlystopping_kwargs)


        # reload best model and evaluate on test
        self.load_model(self.temp_file_folder, file_name="_best_model")

        print("{}: Training complete".format(self.RECOMMENDER_NAME))

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)


    def _run_epoch(self, num_epoch):
        if self.cfm is None:
            raise ValueError('Model not correctly initialized!')
        
        self.cfm._run_epoch(self.data.Train_data)
    
    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")


    def _prepare_model_for_validation(self):
        self.cfm.graph.finalize()

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if items_to_compute is None:
            item_indices = self._item_indices
        else:
            item_indices = items_to_compute

        users_count = len(user_id_array)
        item_scores = -np.ones((users_count, self.n_items)) * np.inf

        users_count_iterator = tqdm(range(users_count)) if self.cfm.verbose else range(users_count)

        for user_index in users_count_iterator:
            user_id = user_id_array[user_index]

            #user_features = self.data.Test_data['X_user'][user_id]
            user_features = self.data.user_map[user_id]
            
            item_score_user = self.cfm.get_scores_per_user(user_features)

            if items_to_compute is not None:
                item_scores[user_index, item_indices] = item_score_user.ravel()[item_indices]
            else:
                item_scores[user_index, :] = item_score_user.ravel()

        return item_scores



    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        #print("Saving model in file '{}'".format(folder_path + file_name))
        # save graph weights
        self.cfm.saver.save(self.cfm.sess, folder_path + file_name)

        # save model params
        data_dict_to_save = {
            'pretrain_flag': self.cfm.pretrain_flag,
            'pretrained_FM_folder_path': self.cfm.pretrained_FM_folder_path,
            'hidden_factor': self.cfm.hidden_factor,
            'batch_size': self.cfm.batch_size,
            'learning_rate': self.cfm.learning_rate,
            'lamda_bilinear': self.cfm.lamda_bilinear,
            'keep': self.cfm.keep,
            'optimizer_type': self.cfm.optimizer_type,
            'batch_norm': self.cfm.batch_norm,
            'verbose': self.cfm.verbose,
            'random_seed': self.cfm.random_seed,
            'gamma_bilinear': self.cfm.gamma_bilinear,
            'lambda_weight': self.cfm.lambda_weight,
            'num_field': self.cfm.num_field,
            'attention_size': self.cfm.attention_size,
            'attentive_pooling': self.cfm.attentive_pooling,
            'regs': self.cfm.regs,
            'net_channel': self.cfm.net_channel,
            'map_mode': self.cfm.map_mode,
            'permutation': self.cfm.permutation
        }
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")


    def load_model(self, folder_path, file_name=None, force_map_mode=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        # initialize the model
        self.cfm = CFM(
            data=self.data,
            user_field_M=self.user_field_M,
            item_field_M=self.item_field_M,
            pretrain_flag=data_dict['pretrain_flag'],
            pretrained_FM_folder_path=data_dict['pretrained_FM_folder_path'],
            hidden_factor=data_dict['hidden_factor'],
            num_field=data_dict['num_field'],
            batch_size=data_dict['batch_size'],
            learning_rate=data_dict['learning_rate'],
            lamda_bilinear=data_dict['lamda_bilinear'],
            keep=data_dict['keep'],
            optimizer_type=data_dict['optimizer_type'],
            batch_norm=data_dict['batch_norm'],
            verbose=data_dict['verbose'],
            regs=data_dict['regs'],
            attention_size=data_dict['attention_size'],
            attentive_pooling=data_dict['attentive_pooling'],
            net_channel=data_dict['net_channel'],
            map_mode=data_dict['map_mode'] if force_map_mode is None else force_map_mode,
            permutation=data_dict['permutation'],
            random_seed=data_dict['random_seed']
        )

        # reload weights
        self.cfm.saver.restore(self.cfm.sess, folder_path + file_name)
        self._print("Loading complete")


