#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/10/19

@author: Federico Parroni
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import argparse

from Base.DataIO import DataIO
from Base.BaseRecommender import BaseRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.BaseTempFolder import BaseTempFolder
from CNN_on_embeddings.IJCAI.CFM_our_interface.FM import FM


#=============== Arguments ===============#
def parse_args():
    parser = argparse.ArgumentParser(description="Run FM.")
    parser.add_argument('--validate', type=bool, default=True,
                        help='Ùse validation or not')
    parser.add_argument('--topk', nargs='?', default=10,
                        help='Topk recommendation list')
    parser.add_argument('--dataset', nargs='?', default='lastfm',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='Keep probility (1-dropout_ratio) for the Bi-Interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 or 1)')

    return parser.parse_args()


class FM_Wrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "FM_Wrapper"

    def __init__(self, URM_train, dataset):
        super().__init__(URM_train)

        self.data = dataset
        self.user_field_M = self.data.user_field_M
        self.item_field_M = self.data.item_field_M
        self._item_indices = np.arange(0, self.n_items, dtype=np.int32)

        self.fm = None
        
    def fit(self, pretrain_flag, hidden_factor, epochs, batch_size, learning_rate, lamda_bilinear,
            keep, optimizer_type, batch_norm, verbose, permutation=None, random_seed=2016,
            temp_file_folder = None, **earlystopping_kwargs):

        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        self.fm = FM(self.data,
                     self.user_field_M,
                     self.item_field_M,
                     pretrain_flag,
                     None,
                     hidden_factor,
                     'log_loss',
                     epochs,
                     batch_size,
                     learning_rate,
                     lamda_bilinear,
                     keep,
                     optimizer_type,
                     batch_norm,
                     verbose,
                     permutation,
                     random_seed)

        self._update_best_model()

        self._train_with_early_stopping(epochs, algorithm_name=self.RECOMMENDER_NAME, **earlystopping_kwargs)

        # reload best model and evaluate on test
        self.load_model(self.temp_file_folder, file_name="_best_model")

        print("{}: Training complete".format(self.RECOMMENDER_NAME))

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)


    def _run_epoch(self, num_epoch):
        if self.fm is None:
            # init the underline object
            raise ValueError('Model not correctly initialized!')

        self.fm._run_epoch(self.data.Train_data)
    
    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")


    def _prepare_model_for_validation(self):
        self.fm.graph.finalize()



    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if items_to_compute is None:
            item_indices = self._item_indices
        else:
            item_indices = items_to_compute

        users_count = len(user_id_array)
        item_scores = -np.ones((users_count, self.n_items)) * np.inf

        for user_index in range(users_count):
            user_id = user_id_array[user_index]

            #user_features = self.data.Test_data['X_user'][user_id]
            user_features = self.data.user_map[user_id]
            
            item_score_user = self.fm.get_scores_per_user(user_features)

            if items_to_compute is not None:
                item_scores[user_index, item_indices] = item_score_user.ravel()
            else:
                item_scores[user_index, :] = item_score_user.ravel()

        return item_scores


    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("Saving model in file '{}'".format(folder_path + file_name))

        # save graph weights
        self.fm.save_session(folder_path + file_name)
        # save model params
        data_dict_to_save = {
            'pretrain_flag': self.fm.pretrain_flag,
            'hidden_factor': self.fm.hidden_factor,
            'loss_type': 'log_loss',
            'epoch': self.fm.epoch,
            'batch_size': self.fm.batch_size,
            'learning_rate': self.fm.learning_rate,
            'lamda_bilinear': self.fm.lamda_bilinear,
            'keep': self.fm.keep,
            'optimizer_type': self.fm.optimizer_type,
            'batch_norm': self.fm.batch_norm,
            'permutation': self.fm.permutation,
            'verbose': self.fm.verbose,
            'random_seed': self.fm.random_seed,
        }
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")
    
    def load_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        # initialize the model
        self.fm = FM(self.data, self.user_field_M, self.item_field_M,
                        data_dict['pretrain_flag'], None, data_dict['hidden_factor'],
                        data_dict['loss_type'], data_dict['epoch'], data_dict['batch_size'], data_dict['learning_rate'],
                        data_dict['lamda_bilinear'], data_dict['keep'], data_dict['optimizer_type'],
                        data_dict['batch_norm'], data_dict['verbose'], data_dict['permutation'], data_dict['random_seed'])
        # reload weights
        self.fm.saver.restore(self.fm.sess, folder_path + file_name)

        self._print("Loading complete")

#
# if __name__ == '__main__':
#     logger = logging.getLogger('mylogger')
#     logger.setLevel(logging.DEBUG)
#     fh = logging.FileHandler('fm_wrapper.log')
#     fh.setLevel(logging.DEBUG)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     # fh.setFormatter(formatter)
#     # ch.setFormatter(formatter)
#     logger.addHandler(fh)
#     logger.addHandler(ch)
#
#     # Data loading
#     args = parse_args()
#     args.path = 'ConvolutionRS/CFM/CFM_our_interface/Data/' if args.validate else 'ConvolutionRS/CFM/CFM_github/Data/'
#     dataset = DatasetCFM(args.path, args.dataset)
#
#     cutoff_list = [10]
#     if args.validate:
#         evaluator_valid = EvaluatorHoldout(dataset.URM_validation, cutoff_list, exclude_seen=True)
#     evaluator_test = EvaluatorHoldout(dataset.URM_test, cutoff_list, exclude_seen=True)
#
#     # permutation
#     permutation = np.arange(args.hidden_factor)
#     np.random.shuffle(permutation)
#     # permutation = None
#
#     if args.verbose > 0:
#         print(
#             "FM: dataset=%s, factors=%d,  #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, optimizer=%s, batch_norm=%d, keep=%.2f"
#             % (args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer,
#                args.batch_norm, args.keep_prob))
#
#     pretrained_FM_folder_path = 'pretrained_models/pretrain-FM-{}-{}'.format(args.dataset, args.hidden_factor)
#     # Training
#     t1 = time.time()
#     model = FM_Wrapper(dataset)
#
#     print("begin train")
#     if args.validate:
#         model.fit(args.pretrain, pretrained_FM_folder_path, args.hidden_factor, args.epoch,
#                     args.batch_size, args.lr, args.lamda, args.keep_prob,
#                     args.optimizer, args.batch_norm, args.verbose, permutation,
#                     validation_every_n=10, lower_validations_allowed=3, stop_on_validation=True,
#                     validation_metric='HIT_RATE', evaluator_object=evaluator_valid)
#     else:
#         model.fit(args.pretrain, pretrained_FM_folder_path, args.hidden_factor, args.epoch,
#                     args.batch_size, args.lr, args.lamda, args.keep_prob,
#                     args.optimizer, args.batch_norm, args.verbose, permutation)
#     print("end train")
#     print('Elapsed time: {}s'.format(time.time() - t1))
#     print()
#
#     # print('Saving model...')
#     # model.save_weights()
#     # print('Done')
#
#     # Evaluation on test
#     print('Evaluating on test...')
#     test_result_dict = evaluator_test.evaluateRecommender(model)
#     print(test_result_dict)
