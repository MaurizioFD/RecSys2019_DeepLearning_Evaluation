# coding=UTF-8
import gc
import time
from time import time

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import Dense, Activation, Flatten, Lambda, Reshape, MaxPooling2D, AveragePooling2D
from keras.layers import Embedding, Input, merge, Conv2D, Multiply, Concatenate, Lambda, Dot
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model

# from LoadTafengDataCnn import load_itemGenres_as_matrix
# from LoadTafengDataCnn import load_negative_file
# from LoadTafengDataCnn import load_rating_file_as_list
# from LoadTafengDataCnn import load_rating_train_as_matrix
# from LoadTafengDataCnn import load_user_attributes
# from evaluateTafengCnn import evaluate_model


def get_train_instances(users_attr_mat, ratings, items_genres_mat, num_negatives=4):
    user_attr_input, item_attr_input, user_id_input, item_id_input, labels = [], [], [], [], []
    num_users, num_items = ratings.shape

    for (u, i) in ratings.keys():
        # positive instance
        # user_vec_input.append(users_vec_mat[u])
        user_attr_input.append(users_attr_mat[u])
        user_id_input.append([u])
        item_id_input.append([i])
        item_attr_input.append(items_genres_mat[i])
        labels.append([1])

        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in ratings:
                j = np.random.randint(num_items)

            # user_vec_input.append(users_vec_mat[u])
            user_attr_input.append(users_attr_mat[u])
            user_id_input.append([u])
            item_id_input.append([j])
            item_attr_input.append(items_genres_mat[j])
            labels.append([0])
    # array_user_vec_input = np.array(user_vec_input)
    array_user_attr_input = np.array(user_attr_input)
    array_user_id_input = np.array(user_id_input)
    array_item_id_input = np.array(item_id_input)
    array_item_attr_input = np.array(item_attr_input)
    array_labels = np.array(labels)

    del user_attr_input, user_id_input, item_id_input, item_attr_input, labels
    gc.collect()

    return array_user_attr_input, array_user_id_input, array_item_attr_input, array_item_id_input, array_labels


def get_model_0(num_users, num_items):
    """
    Model with no convolution
    """
    num_users = num_users + 1
    num_items = num_items + 1

    ########################   attr side   ##################################

    # Input
    user_attr_input = Input(shape=(21,), dtype='float32', name='user_attr_input')
    user_attr_embedding = Dense(8, activation='relu')(user_attr_input)
    user_attr_embedding = Reshape((1, 8))(user_attr_embedding)

    item_sub_class_input = Input(shape=(1,), dtype='float32')
    item_sub_class = Embedding(input_dim=2012, output_dim=3, input_length=1)(item_sub_class_input)
    item_sub_class = Flatten()(item_sub_class)

    item_asset_price_input = Input(shape=(2,), dtype='float32')
    item_asset_price = Dense(5, activation='relu')(item_asset_price_input)

    item_attr_embedding = Concatenate()([item_sub_class, item_asset_price])

    item_attr_embedding = Reshape((8, 1))(item_attr_embedding)

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten()(merge_attr_embedding)
    # merge_attr_embedding = Reshape((8, 8, 1))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = AveragePooling2D((2, 2))(merge_attr_embedding)
    # merge_attr_embedding = Dropout(0.35)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    # merge_attr_embedding = Flatten()(merge_attr_embedding)
    # merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat')

    attr_1 = Dense(16)(merge_attr_embedding_global)
    attr_1 = Activation('relu')(attr_1)
    #    attr_1=BatchNormalization()(attr_1)
    #    attr_1=Dropout(0.2)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten()(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten()(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = Multiply()([user_id_Embedding, item_id_Embedding])
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)

    id_2 = Dense(32)(merge_id_embedding)
    id_2 = Activation('relu')(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = Concatenate()([attr_1, id_2])
    dense_1 = Dense(64)(merge_attr_id_embedding)
    dense_1 = Activation('relu')(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
                     name='topLayer')(dense_1)

    # Final prediction layer
    model = Model(input=[user_attr_input, item_sub_class_input, item_asset_price_input, user_id_input, item_id_input],
                  output=topLayer)

    return model


def get_model_1(num_users, num_items):
    """
    merge_attr_embedding NOT concatenated with merge_attr_embedding_global
    """
    num_users = num_users + 1
    num_items = num_items + 1

    ########################   attr side   ##################################

    # Input
    user_attr_input = Input(shape=(21,), dtype='float32', name='user_attr_input')
    user_attr_embedding = Dense(8, activation='relu')(user_attr_input)
    user_attr_embedding = Reshape((1, 8))(user_attr_embedding)

    item_sub_class_input = Input(shape=(1,), dtype='float32')
    item_sub_class = Embedding(input_dim=2012, output_dim=3, input_length=1)(item_sub_class_input)
    item_sub_class = Flatten()(item_sub_class)

    item_asset_price_input = Input(shape=(2,), dtype='float32')
    item_asset_price = Dense(5, activation='relu')(item_asset_price_input)

    item_attr_embedding = Concatenate()([item_sub_class, item_asset_price])

    item_attr_embedding = Reshape((8, 1))(item_attr_embedding)

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))(
        [user_attr_embedding, item_attr_embedding])

    # merge_attr_embedding_global = Flatten()(merge_attr_embedding)
    merge_attr_embedding = Reshape((8, 8, 1))(merge_attr_embedding)

    merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = AveragePooling2D((2, 2))(merge_attr_embedding)
    # merge_attr_embedding = Dropout(0.35)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten()(merge_attr_embedding)
    # merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat')

    attr_1 = Dense(16)(merge_attr_embedding)
    attr_1 = Activation('relu')(attr_1)
    #    attr_1=BatchNormalization()(attr_1)
    #    attr_1=Dropout(0.2)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten()(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten()(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = Multiply()([user_id_Embedding, item_id_Embedding])
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)

    id_2 = Dense(32)(merge_id_embedding)
    id_2 = Activation('relu')(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = Concatenate()([attr_1, id_2])
    dense_1 = Dense(64)(merge_attr_id_embedding)
    dense_1 = Activation('relu')(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
                     name='topLayer')(dense_1)

    # Final prediction layer
    model = Model(input=[user_attr_input, item_sub_class_input, item_asset_price_input, user_id_input, item_id_input],
                  output=topLayer)

    return model


def get_model_2(num_users, num_items, map_mode='all_map'):
    assert map_mode in ["all_map", "main_diagonal", "off_diagonal"],\
        'Invalid map_mode {}'.format(map_mode)

    num_users = num_users + 1
    num_items = num_items + 1

    permutation_input = Input(shape=(8,8,), dtype=np.float32, name='permutation_matrix')
    ########################   attr side   ##################################

    # Input
    user_attr_input = Input(shape=(21,), dtype='float32', name='user_attr_input')
    user_attr_embedding = Dense(8, activation='relu')(user_attr_input)
    # permutate user_attr_embedding
    user_attr_embedding = Dot(axes=(2,1))([permutation_input, user_attr_embedding])

    # ======
    user_attr_embedding = Reshape((1, 8))(user_attr_embedding)

    item_sub_class_input = Input(shape=(1,), dtype='float32', name='item_sub_class_input')
    item_sub_class = Embedding(input_dim=2012, output_dim=3, input_length=1)(item_sub_class_input)
    item_sub_class = Flatten()(item_sub_class)

    item_asset_price_input = Input(shape=(2,), dtype='float32', name='item_asset_price_input')
    item_asset_price = Dense(5, activation='relu')(item_asset_price_input)

    item_attr_embedding = Concatenate()([item_sub_class, item_asset_price])
    # permutate item_attr_embedding
    item_attr_embedding = Dot(axes=(2,1))([permutation_input, item_attr_embedding])

    # ======
    item_attr_embedding = Reshape((8, 1))(item_attr_embedding)

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten()(merge_attr_embedding)

    # If using only the diagonal, remove everything not in the diagonal
    if map_mode == "main_diagonal":
        print("CoupledCF: Using main diagonal elements.")
        diagonal = Lambda(lambda x: tf.linalg.diag_part(x))(merge_attr_embedding)
        merge_attr_embedding = Lambda(lambda x: tf.linalg.set_diag(K.zeros_like(merge_attr_embedding), x) )(diagonal)

    elif map_mode == "off_diagonal":
        print("CoupledCF: Using off diagonal elements.")
        diagonal = K.zeros_like( Lambda(lambda x: tf.linalg.diag_part(x))(merge_attr_embedding) )
        merge_attr_embedding = Lambda(lambda x: tf.linalg.set_diag(x, diagonal) )(merge_attr_embedding)

    else:
        print("CoupledCF: Using all map elements.")

    merge_attr_embedding = Reshape((8, 8, 1))(merge_attr_embedding)

    merge_attr_embedding = Conv2D(8, (3, 3), name='conv2d_0')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = AveragePooling2D((2, 2))(merge_attr_embedding)
    # merge_attr_embedding = Dropout(0.35)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    merge_attr_embedding = Conv2D(8, (3, 3), name='conv2d_1')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten()(merge_attr_embedding)
    merge_attr_embedding = Concatenate()([merge_attr_embedding, merge_attr_embedding_global])

    attr_1 = Dense(16, name='attr')(merge_attr_embedding)
    attr_1 = Activation('relu')(attr_1)
    #    attr_1=BatchNormalization()(attr_1)
    #    attr_1=Dropout(0.2)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  embeddings_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten()(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  embeddings_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten()(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = Multiply()([user_id_Embedding, item_id_Embedding])
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)

    id_2 = Dense(32, name='merge_id_embedding')(merge_id_embedding)
    id_2 = Activation('relu')(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = Concatenate()([attr_1, id_2])
    dense_1 = Dense(64, name='merge_attr_id_embedding')(merge_attr_id_embedding)
    dense_1 = Activation('relu')(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    topLayer = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform',
                     name='topLayer')(dense_1)

    # Final prediction layer
    model = Model(inputs=[user_attr_input, item_sub_class_input, item_asset_price_input, user_id_input, item_id_input, permutation_input],
                  outputs=topLayer)

    return model


def get_model_3(num_users, num_items):
    #id only (deep CF)

    num_users = num_users + 1
    num_items = num_items + 1

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten()(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten()(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = Multiply()([user_id_Embedding, item_id_Embedding])
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)

    id_2 = Dense(32)(merge_id_embedding)
    id_2 = Activation('relu')(id_2)

    # merge attr_id embedding
    #merge_attr_id_embedding = merge([attr_1, id_2], mode='concat')
    dense_1 = Dense(64)(id_2)
    dense_1 = Activation('relu')(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
                     name='topLayer')(dense_1)

    # Final prediction layer
    model = Model(input=[user_attr_input, item_sub_class_input, item_asset_price_input, user_id_input, item_id_input],
                  output=topLayer)

    return model

def main():
    learning_rate = 0.005
    num_epochs = 30
    verbose = 1
    topK = 10
    evaluation_threads = 1
    num_negatives = 4
    startTime = time()

    # load data
    num_users, users_attr_mat = load_user_attributes()
    num_items, items_genres_mat = load_itemGenres_as_matrix()
    # users_vec_mat = load_user_vectors()
    ratings = load_rating_train_as_matrix()

    # load model
    model = get_model_2(num_users, num_items)

    # compile model
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'mae']
    )
    plot_model(model, show_shapes=True, to_file='tafeng_coupledCF.png')
    model.summary()

    # Training model
    best_hr, best_ndcg = 0, 0
    for epoch in range(num_epochs):
        print('The %d epoch...............................' % (epoch))
        t1 = time()
        # Generate training instances
        user_attr_input, user_id_input, item_attr_input, item_id_input, labels = get_train_instances(users_attr_mat,
                                                                                                     ratings,
                                                                                                     items_genres_mat,
                                                                                                     num_negatives=num_negatives)
        item_sub_class = item_attr_input[:, 0]
        item_asset_price = item_attr_input[:, 1:]

        hist = model.fit([user_attr_input, item_sub_class, item_asset_price, user_id_input, item_id_input],
                         labels, epochs=1,
                         batch_size=256,
                         verbose=1,
                         shuffle=True)
        t2 = time()
        # Evaluation
        if epoch % verbose == 0:
            testRatings = load_rating_file_as_list()
            testNegatives = load_negative_file()
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives,
                                           users_attr_mat, items_genres_mat, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr = hr
                if hr > 0.6:
                    model.save_weights('Pretrain/tafeng_coupledCF_neg_%d_hr_%.4f_ndcg_%.4f.h5' %
                                       (num_negatives, hr, ndcg), overwrite=True)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
    endTime = time()
    print("End. best HR = %.4f, best NDCG = %.4f,time = %.1f s" %
          (best_hr, best_ndcg, endTime - startTime))
    print('HR = %.4f, NDCG = %.4f' % (hr, ndcg))


if __name__ == '__main__':
    main()
