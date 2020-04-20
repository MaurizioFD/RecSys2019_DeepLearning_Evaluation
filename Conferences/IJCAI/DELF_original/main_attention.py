# -*- coding:utf-8 -*-
import functools
import tensorflow as tf
import argparse
import numpy as np
from evaluate_batch import evaluate_model
from Dataset import Dataset
from time import time
import Model.NMF_attention_noAtt
import Model.NMF_attention_MLP
import Model.NMF_attention_EF
import Model.NMF_attention_NSVD
import os
import pickle
import sys
import atexit


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--model', nargs='?', default='MLP',
                        help='Model to use.')
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='movielens1m-paper',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[256,128,64]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--sign', nargs='?', default='1',
                        help='Choose a sign')
    parser.add_argument('--gpu', type=int, default=1,
                        help='Choose a gpu')
    parser.add_argument('--recent_len', type=int, default=5,
                        help='Choose recent length for sequential model')
    return parser.parse_args()


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    num_items = train.shape[1]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            #while train.has_key((u, j)): #python 2.7
            while (u, j) in train: # python3 
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
            # for t in xrange(num_negatives):
            #     j = np.random.randint(num_users)
            #     while train.has_key((j, i)):
            #         j = np.random.randint(num_users)
            #     user_input.append(j)
            #     item_input.append(i)
            #     labels.append(0)
    return user_input, item_input, labels


def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b)
    assert len(a) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


def main():
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    recent_len = args.recent_len
    if dataset == "gowalla":
        topK = 5
    elif dataset == "movielens1m-paper":
        topK = 10
    else:
        raise NotImplementedError
    evaluation_threads = 1  # mp.cpu_count()
    predictive_factors = eval(args.layers)[-1]
    print("MLP arguments: %s " % (args))
    tf.set_random_seed(1234)
    np.random.seed(1234)

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%(args.gpu)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    best_hr, best_ndcg, best_iter = 0, 0, -1
    # best_hr, best_ndcg, best_iter = 0, 0, -1
    best_hr_test, best_ndcg_test = 0,0
    hr_test, ndcg_test = 0, 0


    def exit_handler():
        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
        print("Test performance for this model: HR = %.4f, NDCG = %.4f. " % (hr_test, ndcg_test))

    atexit.register(exit_handler)

    t1 = time()
    dataset = Dataset(args.path + args.dataset, recent_len=recent_len)
    train, validRatings, testRatings, testNegatives = dataset.trainMatrix, \
                                                                    dataset.validRatings, dataset.testRatings, dataset.testNegatives
    validNegatives = np.asarray(testNegatives)[:, :].tolist()
    print (len(validRatings))
    print (len(validRatings[0]))

    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    train_arr = train.toarray()
    input_user = tf.placeholder(tf.int32, [None, 1])
    input_item = tf.placeholder(tf.int32, [None, 1])
    output = tf.placeholder(tf.float32, [None, 1])
    rating_matrix = tf.placeholder(tf.float32, shape=(num_users, num_items))
    user_input, item_input, labels = get_train_instances(train, num_negatives)
    batch_len = len(user_input) // batch_size
    NMFs = {"EF": Model.NMF_attention_EF, 'MLP': Model.NMF_attention_MLP, 'noAtt': Model.NMF_attention_noAtt,
            "NSVD": Model.NMF_attention_NSVD}
    NMF = NMFs[args.model]
    model = NMF.Model(input_user, input_item, output, num_users, num_items, rating_matrix, layers, batch_len)
    tf.summary.histogram("input_user", input_user)
    merged_summary = tf.summary.merge_all()

    saver = tf.train.Saver({"embedding_users": model.embedding_users,
                            "embedding_items": model.embedding_items})
    # var_list_0 = [v for v in tf.global_variables() if "trainable_alpha" in v.name and "Adam" not in v.name]
    # var_list = [v for v in tf.global_variables() if "trainable_alpha" not in v.name and "Adam" not in v.name]
    # saver_model_0 = tf.train.Saver(var_list=var_list_0)
    # saver_model = tf.train.Saver(var_list=var_list)
    saver_model = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    # if os._exists()
    # saver.restore(sess, "Pretrain/MLP_%s_embeddings_%s" % (
    #                         args.dataset, predictive_factors))
    # saver_model_0.restore(sess, "./Pretrain/attention_self_%s_inter1_att" % args.dataset)
    # saver_model.restore(sess, "./Pretrain/attention_self_%s_inter1_mean" % args.dataset)
    # saver_model_2 = tf.train.Saver()

    # saver = tf.train.Saver()
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # # saver.restore(sess, "./Pretrain/fmn")
    # sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./tmp/NMF_fmn")
    writer.add_graph(sess.graph)
    # print ("Begin to evaluate the initial performance...")
    # t1 = time()
    # (hits, ndcgs) = evaluate_model(model, validRatings, validNegatives, topK, evaluation_threads,
    #                                sess, input_user, input_item, rating_matrix, train_arr)
    # hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    # print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    ######best_hr, best_ndcg, best_iter = 0, 0, -1
    # best_hr, best_ndcg, best_iter = 0, 0, -1
    ######best_hr_test, best_ndcg_test = 0,0

    # print "Evaluating on valid set..."
    # (hits, ndcgs) = evaluate_model(model, validRatings, testNegatives, topK, evaluation_threads,
    #                                sess, input_user, input_item, rating_matrix, train_arr)
    # hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    # print('HR = %.4f, NDCG = %.4f'
    #       % (hr, ndcg))

    # training
    lr = 0.01
    loss_last = np.inf
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        user_input, item_input, labels = unison_shuffled_copies(
            np.asarray(user_input), np.asarray(item_input), np.asarray(labels))
        print ("Begin training...")
        # Training
        print (len(user_input))
        batch_len = len(user_input) // batch_size
        for batch in range(batch_size):
            # x, item_embedding_rep, embedding_items_rep, att_embedding_items = sess.run(model.predict,
            #                    feed_dict={input_user: np.expand_dims(user_input, axis=1)[
            #                                           batch * batch_len:(batch + 1) * batch_len],
            #                               input_item: np.expand_dims(item_input, axis=1)[
            #                                           batch * batch_len:(batch + 1) * batch_len],
            #                               rating_matrix: train_arr})
            _, loss = sess.run(model.optimize,
                               feed_dict={input_user: np.expand_dims(user_input, axis=1)[
                                                      batch * batch_len:(batch + 1) * batch_len],
                                          input_item: np.expand_dims(item_input, axis=1)[
                                                      batch * batch_len:(batch + 1) * batch_len],
                                          output: np.expand_dims(labels, axis=1)[
                                                  batch * batch_len:(batch + 1) * batch_len],
                                          rating_matrix: train_arr})
            if batch % 16 == 0:
                print ("epoch %d, batch %d/%d, %.2f" % (epoch, batch, batch_size, loss))
        if loss_last < loss - 0.01:
            lr /= 10.
            loss_last = loss
        t2 = time()
        print ("Training succeeded!")

        if epoch % 10 == 0 and epoch != 0:
            print ("Evaluating on valid set...")
            (hits, ndcgs) = evaluate_model(model, validRatings, validNegatives, topK, evaluation_threads,
                                           sess, input_user, input_item, rating_matrix, train_arr)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, time() - t2))
            # if not args.dataset in ['ml-1m','yelp']:
            if epoch % 10 == 0:
                print ("Evaluating on test set...")
                t2 = time()
                (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads,
                                               sess, input_user, input_item, rating_matrix, train_arr)
                hr_test, ndcg_test = np.array(hits).mean(), np.array(ndcgs).mean()
                if (hr_test + ndcg_test) / 2. > (best_hr_test + best_ndcg_test) / 2.:
                    best_hr_test, best_ndcg_test = hr_test, ndcg_test
                    best_iter_test = epoch

                print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]'
                      % (epoch, t2 - t1, hr_test, ndcg_test, time() - t2))
            if (hr + ndcg) / 2. > (best_hr + best_ndcg) / 2.:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                # best_hr_test_byvalid, best_ndcg_test_byvalid = hr_test, ndcg_test
                if args.out > 0:
                    save_path = saver_model.save(sess, "Pretrain/DENCF_%s_%s_%s" % (args.model,
                                                                                    args.dataset, predictive_factors))
                    print("Model saved in file: %s" % save_path)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if not args.dataset == 'ml-1m':
        # print("Test performance in Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (
        #     best_iter, best_hr_test_byvalid, best_ndcg_test_byvalid))
        print("Best test Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter_test, best_hr_test, best_ndcg_test))

        #
        #     if epoch % verbose == 0:
        #         print "Evaluating on valid set..."
        #         # watch = sess.run(model.watch_variables)
        #         # print watch
        #         (hits, ndcgs) = evaluate_model(model, validRatings, testNegatives, topK, evaluation_threads,
        #                                        sess, input_user, input_item, rating_matrix, train_arr)
        #         hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        #         print('Valid for iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]'
        #               % (epoch, t2 - t1, hr, ndcg, time() - t2))
        #         if (hr + ndcg) / 2. > (best_hr + best_ndcg) / 2.:
        #             best_hr, best_ndcg, best_iter = hr, ndcg, epoch
        #             if args.out > 0:
        #                 # save_path = saver_model_0.save(sess, "./Pretrain/attention_self_%s_inter1_att" %
        #                 #                              (args.dataset))
        #                 save_path = saver_model.save(sess, "/Dataset/weiyu/Pretrain/DENCF_EF_%s_%s" % (
        #                     args.dataset, predictive_factors))
        #                 print("Model saved in file: %s" % save_path)
        #             print "Evaluating on test set..."
        #             t2 = time()
        #             (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads,
        #                                            sess, input_user, input_item, rating_matrix, train_arr)
        #             hr_test, ndcg_test = np.array(hits).mean(), np.array(ndcgs).mean()
        #             print('Testing for iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]'
        #                   % (epoch, t2 - t1, hr_test, ndcg_test, time() - t2))
        #
        # print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))


if __name__ == '__main__':
    main()
