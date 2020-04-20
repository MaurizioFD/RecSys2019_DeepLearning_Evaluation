import functools
import tensorflow as tf
import argparse
import numpy as np
from evaluate_batch_MLP import evaluate_model
from Dataset import Dataset
from time import time
import Model.NMF as NMF
import pandas as pd
import pickle
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='movielens%sample',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
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
        for t in xrange(num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
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
    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("MLP arguments: %s " % (args))
    tf.set_random_seed(1234)
    np.random.seed(1234)

    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    train_arr = train.toarray()
    input_user = tf.placeholder(tf.int32, [None, 1])
    input_item = tf.placeholder(tf.int32, [None, 1])
    output = tf.placeholder(tf.float32, [None, 1])
    rating_matrix = tf.placeholder(tf.float32, shape=(num_users, num_items))
    model = NMF.Model(input_user, input_item, output, num_users, num_items, layers)
    tf.summary.histogram("input_user", input_user)
    merged_summary = tf.summary.merge_all()

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, "./Pretrain/MLP")

    print ("Begin to evaluate the performance...")
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads,
                                   sess, input_user, input_item)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Eval overall performance: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    movielens_sample_df = pd.read_csv("./Data/movielens%sample.train.rating",
                                      names=["User", "Item", "time"], delimiter='\t', low_memory=False)
    gbuser = movielens_sample_df.groupby('User')["Item"].count()

    step = 10
    start = gbuser.min()
    # hits_list, ndcgs_list=[],[]
    record = [[], []]
    for i in xrange(1, 100):
        if start>200:
            step = 100
        if start>1000:
            step = 1000
        if start>4000:
            break
        t1 = time()
        index = gbuser.index[(gbuser >= start) & (gbuser < start + step)].tolist()
        # testRatings_this = [testRatings[i] for i in index]
        # testNegatives_this = [testNegatives[i] for i in index]
        # (hits, ndcgs) = evaluate_model(model, testRatings_this, testNegatives_this, topK, evaluation_threads,
        #                                sess, input_user, input_item, rating_matrix, train_arr)
        # hits_list.extend(hits)
        # ndcgs_list.extend(ndcgs)
        # hr_all = np.array(hits_list).mean()
        # ndcg_all = np.array(ndcgs_list).mean()
        hits_this = [hits[i] for i in index]
        ndcgs_this = [ndcgs[i] for i in index]
        hr, ndcg = np.array(hits_this).mean(), np.array(ndcgs_this).mean()
        print('Eval for user with %d~%d items: HR = %.4f, NDCG = %.4f [%.1f], num_of_users: %d'
              % (start, start + step, hr, ndcg, time() - t1, len(index)))
        # print('Eval for user with %d~%d items: HR = %.4f, NDCG = %.4f'
        #       % (gbuser.min(), start + step, hr_all, ndcg_all))
        record[0].append(hr)
        record[1].append(ndcg)
        start += step

    np.save("Record_MLP.npy", np.asarray(record))
    t1 = time()
    index = gbuser.index[(gbuser >= start)].tolist()
    # testRatings_this = [testRatings[i] for i in index]
    # testNegatives_this = [testNegatives[i] for i in index]
    # (hits, ndcgs) = evaluate_model(model, testRatings_this, testNegatives_this, topK, evaluation_threads,
    #                                sess, input_user, input_item, rating_matrix, train_arr)
    # hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    hits_this = [hits[i] for i in index]
    ndcgs_this = [ndcgs[i] for i in index]
    hr, ndcg = np.array(hits_this).mean(), np.array(ndcgs_this).mean()
    print('Eval for user with %d~%d items: HR = %.4f, NDCG = %.4f [%.1f]'
          % (start, gbuser.max(), hr, ndcg, time() - t1))


if __name__ == '__main__':
    main()
