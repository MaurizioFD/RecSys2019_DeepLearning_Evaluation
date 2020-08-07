#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""


from Base.BaseRecommender import BaseRecommender
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping


import numpy as np
import scipy.sparse as sps
from Base.DataIO import DataIO
import os
from keras.regularizers import l1, l2
from keras.models import Model, load_model, save_model, clone_model
from keras.layers import Embedding, Input, Dense, Reshape, Flatten, Dropout, Concatenate, Multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.backend import clear_session


def MLP_get_model(num_users, num_items, layers = [20,10], reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = 'user_embedding',
                                   embeddings_initializer = 'random_normal', embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'item_embedding',
                                   embeddings_initializer = 'random_normal', embeddings_regularizer = l2(reg_layers[0]), input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))

    # The 0-th layer is the concatenation of embedding layers
    vector = Concatenate()([user_latent, item_latent])

    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer = l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(vector)

    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)

    return model


def GMF_get_model(num_users, num_items, latent_dim, regs=[0,0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  embeddings_initializer = 'random_normal', embeddings_regularizer = l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  embeddings_initializer = 'random_normal', embeddings_regularizer = l2(regs[1]), input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))

    # Element-wise product of user and item embeddings
    predict_vector = Multiply()([user_latent, item_latent])

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)

    model = Model(inputs=[user_input, item_input],
                outputs=prediction)

    return model


def NeuCF_get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0.0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  embeddings_initializer = 'random_normal', embeddings_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  embeddings_initializer = 'random_normal', embeddings_regularizer = l2(reg_mf), input_length=1)

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = "mlp_embedding_user",
                                   embeddings_initializer = 'random_normal', embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'mlp_embedding_item',
                                   embeddings_initializer = 'random_normal', embeddings_regularizer = l2(reg_layers[0]), input_length=1)

    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent]) # element-wise multiply

    # MLP part
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    predict_vector = Concatenate()([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = "prediction")(predict_vector)

    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)

    return model



def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])
    return model


def get_train_instances(train, num_negatives, num_items):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():#train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels



def set_learner(model, learning_rate, learner):

    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

    return model


def deep_clone_model(source_model):

    destination_model = clone_model(source_model)
    destination_model.set_weights(source_model.get_weights())

    return destination_model



class NeuMF_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping):


    RECOMMENDER_NAME = "NeuMF_RecommenderWrapper"

    def __init__(self, URM_train):
        super(NeuMF_RecommenderWrapper, self).__init__(URM_train)

        self._train = sps.dok_matrix(self.URM_train)
        self.n_users, self.n_items = self.URM_train.shape

        self._item_indices = np.arange(0, self.n_items, dtype=np.int)
        self._user_ones_vector = np.ones_like(self._item_indices)


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            # The prediction requires a list of two arrays user_id, item_id of equal length
            # To compute the recommendations for a single user, we must provide its index as many times as the
            # number of items
            item_score_user = self.model.predict([self._user_ones_vector*user_id, self._item_indices],
                                                 batch_size=100, verbose=0)


            if items_to_compute is not None:
                item_scores[user_index, items_to_compute] = item_score_user.ravel()[items_to_compute]
            else:
                item_scores[user_index, :] = item_score_user.ravel()


        return item_scores

    def get_USER_embeddings(self):
        return self.model.get_weights()[2]


    def get_ITEM_embeddings(self):
        return self.model.get_weights()[3]

    def get_early_stopping_final_epochs_dict(self):
        """
        This function returns a dictionary to be used as optimal parameters in the .fit() function
        It provides the flexibility to deal with multiple early-stopping in a single algorithm
        e.g. in NeuMF there are three model components each with its own optimal number of epochs
        the return dict would be {"epochs": epochs_best_neumf, "epochs_gmf": epochs_best_gmf, "epochs_mlp": epochs_best_mlp}
        :return:
        """

        return {"epochs": self.epochs_best, "epochs_gmf": self.epochs_best_gmf, "epochs_mlp": self.epochs_best_mlp}




    def fit(self,
            epochs = 100,
            epochs_gmf=100,
            epochs_mlp=100,
            batch_size = 256,
            num_factors = 8,
            layers = [64,32,16,8],
            reg_mf = 0.0,
            reg_layers = [0,0,0,0],
            num_negatives = 4,
            learning_rate = 1e-3,
            learning_rate_pretrain = 1e-3,
            learner = 'sgd',
            learner_pretrain = 'adam',
            pretrain = True,
            root_folder_pretrain = None,
            **earlystopping_kwargs):
        """

        :param epochs:
        :param batch_size:
        :param num_factors: Embedding size of MF model
        :param layers: MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.
        :param reg_mf: Regularization for MF embeddings.
        :param reg_layers: Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.
        :param num_negatives: Number of negative instances to pair with a positive instance.
        :param learning_rate:
        :param learning_rate_pretrain:
        :param learner: adagrad, adam, rmsprop, sgd
        :param learner_pretrain: adagrad, adam, rmsprop, sgd
        :param root_folder_pretrain: Specify the pretrain model folder where to save MF and MLP for MF part.
        :param do_pretrain:
        :return:
        """

        clear_session()

        self.batch_size = batch_size
        self.mf_dim = num_factors
        self.layers = layers.copy()
        self.reg_mf = reg_mf
        self.reg_layers = reg_layers.copy()
        self.num_negatives = num_negatives

        assert learner in ["adagrad", "adam", "rmsprop", "sgd"]
        assert learner_pretrain in ["adagrad", "adam", "rmsprop", "sgd"]

        self.pretrain = pretrain

        if self.pretrain:

            if root_folder_pretrain is not None:
                print("NeuMF_RecommenderWrapper: pretrained models will be saved in '{}'".format(root_folder_pretrain))

                # If directory does not exist, create
                if not os.path.exists(root_folder_pretrain):
                    os.makedirs(root_folder_pretrain)

            print("NeuMF_RecommenderWrapper: root_folder_pretrain not provided, pretrained models will not be saved")

            print("NeuMF_RecommenderWrapper: Pretraining GMF...")

            self.model = GMF_get_model(self.n_users, self.n_items, self.mf_dim)
            self.model = set_learner(self.model, learning_rate_pretrain, learner_pretrain)

            self._best_model = deep_clone_model(self.model)

            self._train_with_early_stopping(epochs_gmf,
                                            algorithm_name = self.RECOMMENDER_NAME,
                                            **earlystopping_kwargs)

            self.epochs_best_gmf = self.epochs_best

            if root_folder_pretrain is not None:
                model_out_file = "GMF_factors_{}_pretrain".format(self.mf_dim)
                self._best_model.save_weights(root_folder_pretrain + model_out_file, overwrite=True)

            self.gmf_model = deep_clone_model(self._best_model)



            print("NeuMF_RecommenderWrapper: Pretraining MLP...")

            self.model = MLP_get_model(self.n_users, self.n_items, self.layers, self.reg_layers)
            self.model = set_learner(self.model, learning_rate_pretrain, learner_pretrain)

            self._best_model = deep_clone_model(self.model)

            self._train_with_early_stopping(epochs_mlp,
                                            algorithm_name = self.RECOMMENDER_NAME,
                                            **earlystopping_kwargs)

            self.epochs_best_mlp = self.epochs_best

            if root_folder_pretrain is not None:
                model_out_file = "MLP_layers_{}_reg_layers_{}_pretrain".format(self.layers, reg_layers)
                self._best_model.save_weights(root_folder_pretrain + model_out_file, overwrite=True)

            self.mlp_model = deep_clone_model(self._best_model)





        # Build model
        self.model = NeuCF_get_model(self.n_users, self.n_items, self.mf_dim, self.layers, self.reg_layers, self.reg_mf)
        self.model = set_learner(self.model, learning_rate, learner)


        # Load pretrain model
        if pretrain:
            self.model = load_pretrain_model(self.model, self.gmf_model, self.mlp_model, len(layers))
            print("NeuMF_RecommenderWrapper: Load pretrained GMF and MLP models.")


        print("NeuMF_RecommenderWrapper: Training NeuCF...")

        self._best_model = deep_clone_model(self.model)

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)


        self._print("Training complete")

        self.model = deep_clone_model(self._best_model)




    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        # Keras only clones the structure of the model, not the weights
        self._best_model = deep_clone_model(self.model)


    def _run_epoch(self, currentEpoch):

        # Generate training instances
        user_input, item_input, labels = get_train_instances(self._train, self.num_negatives, self.n_items)

        # Training
        hist = self.model.fit([np.array(user_input).astype(np.int32),
                               np.array(item_input).astype(np.int32)], #input
                         np.array(labels).astype(np.int32), # labels
                         batch_size=self.batch_size, epochs=1, verbose=0, shuffle=True)

        print("NeuMF_RecommenderWrapper: Epoch {}, loss {:.2E}".format(currentEpoch+1, hist.history['loss'][0]))




















    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        self.model.save_weights(folder_path + file_name + "_weights", overwrite=True)

        data_dict_to_save = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "mf_dim": self.mf_dim,
            "layers": self.layers,
            "reg_layers": self.reg_layers,
            "reg_mf": self.reg_mf,
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)


        self._print("Saving complete")




    def load_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])

        clear_session()

        self.model = NeuCF_get_model(self.n_users, self.n_items, self.mf_dim, self.layers, self.reg_layers, self.reg_mf)
        self.model.load_weights(folder_path + file_name + "_weights")


        self._print("Loading complete")

