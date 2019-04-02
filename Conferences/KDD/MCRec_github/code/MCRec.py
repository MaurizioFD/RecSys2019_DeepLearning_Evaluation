import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

import numpy as np
import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Dense, Lambda, Activation, LSTM, Reshape, Conv1D, GlobalMaxPooling1D, Dropout
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, concatenate, RepeatVector, multiply#, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adagrad, Adam, SGD, RMSprop, Nadam
from keras.regularizers import l2
from Conferences.KDD.MCRec_github.code.Dataset import Dataset
from Conferences.KDD.MCRec_github.code.evaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse
import scipy.sparse as sp
import gc

def slice(x, index):
    return x[:, index, :, :]


def slice_2(x, index):
    return x[:, index, :]

def path_attention(user_latent, item_latent, path_latent, latent_size, att_size, path_attention_layer_1, path_attention_layer_2, path_name):
    #user_latent (batch_size, latent_size)
    #item_latent (batch_size, latent_size)
    #path_latent (batch_size, path_num, mp_latent_size)
    latent_size = user_latent.shape[1].value
    path_num, path_latent_size = path_latent.shape[1].value, path_latent.shape[2].value

    path = Lambda(slice_2, output_shape=(path_latent_size,), arguments={'index':0})(path_latent)
    inputs = concatenate([user_latent, item_latent, path])
    output = (path_attention_layer_1(inputs))
    output = (path_attention_layer_2(output))
    for i in range(1, path_num):
        path = Lambda(slice_2, output_shape=(path_latent_size,), arguments={'index':i})(path_latent)
        inputs = concatenate([user_latent, item_latent, path])
        tmp_output = (path_attention_layer_1(inputs))
        tmp_output = (path_attention_layer_2(tmp_output))
        output = concatenate([output, tmp_output])
    
    
    atten = Lambda(lambda x : K.softmax(x), name = '%s_attention_softmax'%path_name)(output)
    output = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([path_latent, atten])
    return output

def get_umtmum_embedding(umtmum_input, path_num, timestamps, length, user_latent, item_latent, path_attention_layer_1, path_attention_layer_2):
    conv_umtmum = Conv1D(filters = 128,
                       kernel_size = 4,
                       activation = 'relu',
                       kernel_regularizer = l2(0.0),
                       kernel_initializer = 'glorot_uniform',
                       padding = 'valid',
                       strides = 1,
                       name = 'umtmum_conv')

    path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index':0})(umtmum_input)
    output = conv_umtmum(path_input)
    output = GlobalMaxPooling1D()(output)
    output = Dropout(0.5)(output)

    for i in range(1, path_num):
        path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index':i})(umtmum_input)
        tmp_output = GlobalMaxPooling1D()(conv_umtmum(path_input))
        tmp_output = Dropout(0.5)(tmp_output)
        output = concatenate([output, tmp_output])
    
    output = Reshape((path_num, 128))(output)
    #output = path_attention(user_latent, item_latent, output, 128, 64, path_attention_layer_1, path_attention_layer_2, 'umtmum')
    output = GlobalMaxPooling1D()(output)
    return output

def get_umtm_embedding(umtm_input, path_num, timestamps, length, user_latent, item_latent, path_attention_layer_1, path_attention_layer_2):
    conv_umtm = Conv1D(filters = 128,
                       kernel_size = 4,
                       activation = 'relu',
                       kernel_regularizer = l2(0.0),
                       kernel_initializer = 'glorot_uniform',
                       padding = 'valid',
                       strides = 1,
                       name = 'umtm_conv')

    path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index':0})(umtm_input)
    output = GlobalMaxPooling1D()(conv_umtm(path_input))
    output = Dropout(0.5)(output)

    for i in range(1, path_num):
        path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index':i})(umtm_input)
        tmp_output = GlobalMaxPooling1D()(conv_umtm(path_input))
        tmp_output = Dropout(0.5)(tmp_output)
        output = concatenate([output, tmp_output])
    
    output = Reshape((path_num, 128))(output)
    #output = path_attention(user_latent, item_latent, output, 128, 64, path_attention_layer_1, path_attention_layer_2, 'umtm')
    output = GlobalMaxPooling1D()(output)
    return output
    
def get_umum_embedding(umum_input, path_num, timestamps, length, user_latent, item_latent, path_attention_layer_1, path_attention_layer_2):
    conv_umum = Conv1D(filters = 128,
                       kernel_size = 4,
                       activation = 'relu',
                       kernel_regularizer = l2(0.0),
                       kernel_initializer = 'glorot_uniform',
                       padding = 'valid',
                       strides = 1,
                       name = 'umum_conv')

    path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index':0})(umum_input)
    output = GlobalMaxPooling1D()(conv_umum(path_input))
    output = Dropout(0.5)(output)

    for i in range(1, path_num):
        path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index':i})(umum_input)
        tmp_output = GlobalMaxPooling1D()(conv_umum(path_input))
        tmp_output = Dropout(0.5)(tmp_output)
        output = concatenate([output, tmp_output])
    
    
    output = Reshape((path_num, 128))(output)
    #output = path_attention(user_latent, item_latent, output, 128, 64, path_attention_layer_1, path_attention_layer_2, 'umum')
    output = GlobalMaxPooling1D()(output)
    return output

def get_uuum_embedding(umum_input, path_num, timestamps, length, user_latent, item_latent, path_attention_layer_1, path_attention_layer_2):
    conv_umum = Conv1D(filters = 128,
                       kernel_size = 4,
                       activation = 'relu',
                       kernel_regularizer = l2(0.0),
                       kernel_initializer = 'glorot_uniform',
                       padding = 'valid',
                       strides = 1,
                       name = 'uuum_conv')

    path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index':0})(umum_input)
    output = GlobalMaxPooling1D()(conv_umum(path_input))
    output = Dropout(0.5)(output)

    for i in range(1, path_num):
        path_input = Lambda(slice, output_shape=(timestamps, length), arguments={'index':i})(umum_input)
        tmp_output = GlobalMaxPooling1D()(conv_umum(path_input))
        tmp_output = Dropout(0.5)(tmp_output)
        output = concatenate([output, tmp_output])
    
    
    output = Reshape((path_num, 128))(output)
    #output = path_attention(user_latent, item_latent, output, 128, 64, path_attention_layer_1, path_attention_layer_2, 'uuum')
    output = GlobalMaxPooling1D()(output)
    return output


def metapath_attention(user_latent, item_latent, metapath_latent, latent_size, att_size):
    #user_latent (batch_size, latent_size)
    #item_latent (batch_size, latent_size)
    #metapath_latent (batch_size, path_num, mp_latent_size)
    #print user_latent.shape
    latent_size = user_latent.shape[1].value
    path_num, mp_latent_size = metapath_latent.shape[1].value, metapath_latent.shape[2].value
    dense_layer_1 = Dense(att_size,
                        activation = 'relu',
                        kernel_initializer = 'glorot_normal',
                        kernel_regularizer = l2(0.001),
                        name = 'metapath_attention_layer_1')
    
    dense_layer_2 = Dense(1,
                          activation = 'relu',
                          kernel_initializer = 'glorot_normal',
                          kernel_regularizer =l2(0.001),
                          name = 'metapath_attention_layer_2')

    metapath = Lambda(slice_2, output_shape=(mp_latent_size,), arguments={'index':0})(metapath_latent)
    inputs = concatenate([user_latent, item_latent, metapath])
    output = (dense_layer_1(inputs))
    output = (dense_layer_2(output))
    for i in range(1, path_num):
        metapath = Lambda(slice_2, output_shape=(mp_latent_size,), arguments={'index':i})(metapath_latent)
        inputs = concatenate([user_latent, item_latent, metapath])
        tmp_output = (dense_layer_1(inputs))
        tmp_output = (dense_layer_2(tmp_output))
        output = concatenate([output, tmp_output])
    
    
    atten = Lambda(lambda x : K.softmax(x), name = 'metapath_attention_softmax')(output)
    output = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], -1), 1))([metapath_latent, atten])
    return output


def user_attention(user_latent, path_output):
    latent_size = user_latent.shape[1].value
    
    inputs = concatenate([user_latent, path_output])
    output = Dense(latent_size,
                   activation = 'relu',
                   kernel_initializer = 'glorot_normal',
                   kernel_regularizer =l2(0.001),
                   name = 'user_attention_layer')(inputs)
    atten = Lambda(lambda x : K.softmax(x), name = 'user_attention_softmax')(output)
    output = multiply([user_latent, atten])
    return output
    
def item_attention(item_latent, path_output):
    latent_size = item_latent.shape[1].value
    
    inputs = concatenate([item_latent, path_output])
    output = Dense(latent_size,
                   activation = 'relu',
                   kernel_initializer = 'glorot_normal',
                   kernel_regularizer =l2(0.001),
                   name = 'item_attention_layer')(inputs)
    atten = Lambda(lambda x : K.softmax(x), name = 'item_attention_softmax')(output)
    output = multiply([item_latent, atten])
    return output
    

def get_model(usize, isize, path_nums, timestamps, length, layers = [20, 10], reg_layers = [0, 0], latent_dim = 40, reg_latent = 0):
    user_input = Input(shape = (1,), dtype = 'int32', name = 'user_input', sparse = False)
    item_input = Input(shape = (1,), dtype = 'int32', name = 'item_input', sparse = False)
    umtm_input = Input(shape = (path_nums[0], timestamps[0], length,), dtype = 'float32', name = 'umtm_input')
    umum_input = Input(shape = (path_nums[1], timestamps[1], length,), dtype = 'float32', name = 'umum_input') 
    umtmum_input = Input(shape = (path_nums[2], timestamps[2], length,), dtype = 'float32', name = 'umtmum_input')
    uuum_input = Input(shape = (path_nums[3], timestamps[3], length, ), dtype = 'float32', name = 'uuum_input')
    Embedding_User_Feedback = Embedding(input_dim = usize, 
                                    output_dim = latent_dim,
                                    input_length = 1,
                                    embeddings_initializer = 'glorot_normal',
                                    name = 'user_feedback_embedding')
    
    Embedding_Item_Feedback = Embedding(input_dim = isize, 
                                    output_dim = latent_dim,
                                    input_length = 1,
                                    embeddings_initializer = 'glorot_normal',
                                    name = 'item_feedback_embedding')
    user_latent = Reshape((latent_dim,))(Flatten()(Embedding_User_Feedback(user_input)))
    item_latent = Reshape((latent_dim,))(Flatten()(Embedding_Item_Feedback(item_input)))
    
    
    path_attention_layer_1 = Dense(128,
                                   activation = 'relu',
                                   kernel_regularizer = l2(0.001),
                                   kernel_initializer = 'glorot_normal',
                                   name = 'path_attention_layer_1')
    
    path_attention_layer_2 = Dense(1,
                                   activation = 'relu',
                                   kernel_regularizer = l2(0.001),
                                   kernel_initializer = 'glorot_normal',
                                   name = 'path_attention_layer_2')
    
    umtm_latent = get_umtm_embedding(umtm_input, path_nums[0], timestamps[0], length, user_latent, item_latent, path_attention_layer_1, path_attention_layer_2)
    umum_latent = get_umum_embedding(umum_input, path_nums[1], timestamps[1], length, user_latent, item_latent, path_attention_layer_1, path_attention_layer_2)
    umtmum_latent = get_umtmum_embedding(umtmum_input, path_nums[2], timestamps[2], length, user_latent, item_latent, path_attention_layer_1, path_attention_layer_2)
    uuum_latent = get_uuum_embedding(uuum_input, path_nums[3], timestamps[3], length, user_latent, item_latent, path_attention_layer_1, path_attention_layer_2)
    
    path_output = concatenate([umtm_latent, umum_latent, umtmum_latent, uuum_latent])
    path_output = Reshape((4, 128))(path_output)
    path_output = metapath_attention(user_latent, item_latent, path_output, latent_dim, 128) 
    
    user_atten = user_attention(user_latent, path_output)
    item_atten = item_attention(item_latent, path_output)
     
    output = concatenate([user_atten, path_output, item_atten])
    for idx in range(0, len(layers)):
        layer = Dense(layers[idx],
                      kernel_regularizer = l2(0.001),
                      kernel_initializer = 'glorot_normal',
                      activation = 'relu',
                      name = 'item_layer%d' % idx)
        output = layer(output)
    
    #user_output = concatenate([user_atten, path_output])
    #for idx in xrange(0, len(layers)):
    #    layer = Dense(layers[idx], 
    #                  kernel_regularizer = l2(0.001),
    #                  kernel_initializer = 'glorot_normal',
    #                  activation = 'relu',
    #                  name = 'user_layer%d' % idx)
    #    user_output = layer(user_output)
                                
    #item_output = concatenate([path_output, item_atten])
    #for idx in xrange(0, len(layers)):
    #    layer = Dense(layers[idx],
    #                  kernel_regularizer = l2(0.001),
    #                  kernel_initializer = 'glorot_normal',
    #                  activation = 'relu',
    #                  name = 'item_layer%d' % idx)
    #item_output = layer(item_output)
    
    #output = concatenate([user_output, item_output])

    print('output.shape = ', output.shape)
    prediction_layer = Dense(1, 
                       activation = 'sigmoid',
                       kernel_initializer = 'lecun_normal',
                       name = 'prediction')

    prediction = prediction_layer(output)
    model = Model(inputs = [user_input, item_input, umtm_input, umum_input, umtmum_input, uuum_input], outputs = [prediction])

    return model

def get_train_instances(user_feature, item_feature, type_feature, path_umtm, path_umum, path_umtmum, path_uuum, path_nums, timestamps, train_list, num_negatives, batch_size, shuffle = True):
    num_batches_per_epoch = int((len(train_list) - 1) / batch_size) + 1
    
    def data_generator():
        data_size = len(train_list)
        while True:
            if shuffle == True:
                np.random.shuffle(train_list)
            
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                k = 0
                _user_input = np.zeros((batch_size * (num_negatives + 1),))
                _item_input = np.zeros((batch_size * (num_negatives + 1),))
                _umtm_input = np.zeros((batch_size * (num_negatives + 1), path_nums[0], timestamps[0], 64))
                _umum_input = np.zeros((batch_size * (num_negatives + 1), path_nums[1], timestamps[1], 64))
                _umtmum_input = np.zeros((batch_size * (num_negatives + 1), path_nums[2], timestamps[2], 64))
                _uuum_input = np.zeros((batch_size * (num_negatives + 1), path_nums[3], timestamps[3], 64))
                _labels = np.zeros(batch_size * (num_negatives + 1))
                
                for u, i in train_list[start_index : end_index]:

                    _user_input[k] = u
                    _item_input[k] = i

                    if (u, i) in path_umtm:
                        for p_i in range(len(path_umtm[(u, i)])):
                            for p_j in range(len(path_umtm[(u, i)][p_i])):
                                type_id = path_umtm[(u, i)][p_i][p_j][0]
                                index = path_umtm[(u, i)][p_i][p_j][1]
                                if type_id == 1 :
                                    _umtm_input[k][p_i][p_j] = user_feature[index]
                                elif type_id == 2 :
                                    _umtm_input[k][p_i][p_j] = item_feature[index]
                                elif type_id == 3 :
                                    _umtm_input[k][p_i][p_j] = type_feature[index]

                    if (u, i) in path_umum:
                        for p_i in range(len(path_umum[(u, i)])):
                            for p_j in range(len(path_umum[(u, i)][p_i])):
                                type_id = path_umum[(u, i)][p_i][p_j][0]
                                index = path_umum[(u, i)][p_i][p_j][1]
                                if type_id == 1 :
                                    _umum_input[k][p_i][p_j] = user_feature[index]
                                elif type_id == 2 :
                                    _umum_input[k][p_i][p_j] = item_feature[index]
                                elif type_id == 3 :
                                    _umum_input[k][p_i][p_j] = type_feature[index]
                        
                    if (u, i) in path_umtmum:
                        for p_i in range(len(path_umtmum[(u, i)])):
                            for p_j in range(len(path_umtmum[(u, i)][p_i])):
                                type_id = path_umtmum[(u, i)][p_i][p_j][0]
                                index = path_umtmum[(u, i)][p_i][p_j][1]
                                if type_id == 1 :
                                    _umtmum_input[k][p_i][p_j] = user_feature[index]
                                elif type_id == 2 :
                                    _umtmum_input[k][p_i][p_j] = item_feature[index]
                                elif type_id == 3 :
                                    _umtmum_input[k][p_i][p_j] = type_feature[index]
                        
                    if (u, i) in path_uuum:
                        for p_i in range(len(path_uuum[(u, i)])):
                            for p_j in range(len(path_uuum[(u, i)][p_i])):
                                type_id = path_uuum[(u, i)][p_i][p_j][0]
                                index = path_uuum[(u, i)][p_i][p_j][1]
                                if type_id == 1 :
                                    _uuum_input[k][p_i][p_j] = user_feature[index]
                                elif type_id == 2 :
                                    _uuum_input[k][p_i][p_j] = item_feature[index]
                                elif type_id == 3 :
                                    _uuum_input[k][p_i][p_j] = type_feature[index]
                    _labels[k] = 1.0
                    k += 1
                    #negative instances
                    for t in range(num_negatives):
                        j = np.random.randint(1, num_items-1)
                        while j in user_item_map[u]:
                            j = np.random.randint(1, num_items-1)
                        
                        _user_input[k] = u
                        _item_input[k] = j
            
                        if (u, j) in path_umtm: 
                            for p_i in range(len(path_umtm[(u, j)])):
                                for p_j in range(len(path_umtm[(u, j)][p_i])):
                                    type_id = path_umtm[(u, j)][p_i][p_j][0]
                                    index = path_umtm[(u, j)][p_i][p_j][1]
                                    if type_id == 1 :
                                        _umtm_input[k][p_i][p_j] = user_feature[index]
                                    elif type_id == 2 :
                                        _umtm_input[k][p_i][p_j] = item_feature[index]
                                    elif type_id == 3 :
                                        _umtm_input[k][p_i][p_j] = type_feature[index]
                            
                        if (u, j) in path_umum: 
                            for p_i in range(len(path_umum[(u, j)])):
                                for p_j in range(len(path_umum[(u, j)][p_i])):
                                    type_id = path_umum[(u, j)][p_i][p_j][0]
                                    index = path_umum[(u, j)][p_i][p_j][1]
                                    if type_id == 1 :
                                        _umum_input[k][p_i][p_j] = user_feature[index]
                                    elif type_id == 2 :
                                        _umum_input[k][p_i][p_j] = item_feature[index]
                                    elif type_id == 3 :
                                        _umum_input[k][p_i][p_j] = type_feature[index]
                        if (u, j) in path_umtmum:
                            for p_i in range(len(path_umtmum[(u, j)])):
                                for p_j in range(len(path_umtmum[(u, j)][p_i])):
                                    type_id = path_umtmum[(u, j)][p_i][p_j][0]
                                    index = path_umtmum[(u, j)][p_i][p_j][1]
                                    if type_id == 1 :
                                        _umtmum_input[k][p_i][p_j] = user_feature[index]
                                    elif type_id == 2 :
                                        _umtmum_input[k][p_i][p_j] = item_feature[index]
                                    elif type_id == 3 :
                                        _umtmum_input[k][p_i][p_j] = type_feature[index]
            
                        if (u, j) in path_uuum:
                            for p_i in range(len(path_uuum[(u, j)])):
                                for p_j in range(len(path_uuum[(u, j)][p_i])):
                                    type_id = path_uuum[(u, j)][p_i][p_j][0]
                                    index = path_uuum[(u, j)][p_i][p_j][1]
                                    if type_id == 1 :
                                        _uuum_input[k][p_i][p_j] = user_feature[index]
                                    elif type_id == 2 :
                                        _uuum_input[k][p_i][p_j] = item_feature[index]
                                    elif type_id == 3 :
                                        _uuum_input[k][p_i][p_j] = type_feature[index]
                        _labels[k] = 0.0
                        k += 1
                yield ([_user_input, _item_input, _umtm_input, _umum_input, _umtmum_input, _uuum_input], _labels)
    return num_batches_per_epoch, data_generator()
    
if __name__ == '__main__':

    dataset = 'ml-100k'
    latent_dim = 128
    reg_latent = 0
    layers = [512, 256, 128, 64]
    reg_layes = [0 ,0, 0, 0]
    learning_rate = 0.001
    epochs = 30
    batch_size = 256
    num_negatives = 4
    learner = 'adam'
    verbose = 1
    out = 0
    evaluation_threads = 1
    topK = 10
    
    print('num_negatives = ', num_negatives)

    t1 = time()
    dataset = Dataset('../data/' + dataset)
    trainMatrix, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    train = dataset.train
    user_item_map = dataset.user_item_map
    item_user_map = dataset.item_user_map
    path_umtm = dataset.path_umtm
    path_umum = dataset.path_umum
    path_umtmum = dataset.path_umtmum
    path_uuum = dataset.path_uuum
    user_feature, item_feature, type_feature = dataset.user_feature, dataset.item_feature, dataset.type_feature
    num_users, num_items = trainMatrix.shape[0], trainMatrix.shape[1]
    path_nums = [dataset.umtm_path_num, dataset.umum_path_num, dataset.umtmum_path_num, dataset.uuum_path_num]
    timestamps = [dataset.umtm_timestamp, dataset.umum_timestamp, dataset.umtmum_timestamp, dataset.uuum_timestamp]
    length = dataset.fea_size

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" % (time()-t1, num_users, num_items, len(train), len(testRatings)))
    print('path nums = ', path_nums)
    print('timestamps = ', timestamps)

    model = get_model(num_users, num_items, path_nums, timestamps, length, layers, reg_layes, latent_dim, reg_latent)
    model.compile(optimizer = Adam(lr = learning_rate, decay = 1e-4),
                  loss = 'binary_crossentropy')
    #model.compile(optimizer = Nadam(),
    #              loss = 'binary_crossentropy')
    

    # Check Init performance
    t1 = time()
    (ps, rs, ndcgs) = evaluate_model(model, user_feature, item_feature, type_feature, num_users, num_items, path_umtm, path_umum, path_umtmum, path_uuum, path_nums, timestamps, length, testRatings, testNegatives, topK, evaluation_threads)
    p, r, ndcg = np.array(ps).mean(), np.array(rs).mean(), np.array(ndcgs).mean()
    print('Init: Precision = %.4f, Recall = %.4f, NDCG = %.4f [%.1f]' %(p, r, ndcg, time()-t1))
    
    best_p = -1
    p_list, r_list, ndcg_list = [], [], []
    print('Begin training....')
    
    for epoch in range(epochs):
        t1 = time()
        
        #Generate training instance
        train_steps, train_batches = get_train_instances(user_feature, item_feature, type_feature, path_umtm, path_umum, path_umtmum, path_uuum, path_nums, timestamps, train, num_negatives, batch_size, True)
        t = time()
        print('[%.1f s] epoch %d train_steps %d' % (t - t1, epoch, train_steps))
        #Training
        hist = model.fit_generator(train_batches,
                                   train_steps,
                                   epochs = 1,
                                   verbose = 0)
        print('training time %.1f s' % (time() - t))
        
        
        
        t2 = time()
        if epoch % verbose == 0:
            (ps, rs, ndcgs) = evaluate_model(model, user_feature, item_feature, type_feature, num_users, num_items, path_umtm, path_umum, path_umtmum, path_uuum, path_nums, timestamps, length, testRatings, testNegatives, topK, evaluation_threads)
            p, r, ndcg, loss = np.array(ps).mean(), np.array(rs).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: Precision = %.4f, Recall = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, p, r, ndcg, loss, time()-t2))

            #if p > best_p:
            #    best_p = p
            #    attention_layer_model = Model(inputs=model.input,  
            #                          outputs = [model.get_layer('user_input').output, model.get_layer('item_input').output, model.get_layer('metapath_attention_softmax').output])
            #    [user_input_output, item_input_output, metapath_attention_output] = attention_layer_model.predict_generator(train_batches, train_steps)
            #    with open('../data/ml-100k.attention_2', 'w') as outfile:
            #        num = user_input_output.shape[0]
            #        for i in range(num):
            #            outfile.write(str(user_input_output[i]) + ',' + str(item_input_output[i]))
            #            for j in range(metapath_attention_output.shape[1]):
            #                outfile.write(' ' + str(metapath_attention_output[i][j]))
            #            outfile.write('\n')
            #    print 'write succeccfully...'
            p_list.append(p)
            r_list.append(r)
            ndcg_list.append(ndcg)
    print("End. Precision = %.4f, Recall = %.4f, NDCG = %.4f. " %(max(p_list), max(r_list), max(ndcg_list)))
