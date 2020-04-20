import functools
import tensorflow as tf
import argparse
import numpy as np
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import Model.NMF_nsvd_pretrain as NMF
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
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

    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    input_user = tf.placeholder(tf.int32, [None, 1])
    input_item = tf.placeholder(tf.int32, [None, 1])
    output = tf.placeholder(tf.float32, [None, 1])
    model = NMF.Model(input_user, input_item, output, num_users, num_items, 32, train)

    # tf.reset_default_graph()
    saver = tf.train.Saver({"embed_users": model.embedding_users,
                            "embed_items": model.embedding_items})
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, "./Pretrain/NMF_embed.ckpt")
    print ("Begin to evaluate the initial performance...")
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads,
                                   sess, input_user, input_item)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1

    # training
    for epoch in xrange(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        user_input, item_input, labels = unison_shuffled_copies(
            np.asarray(user_input), np.asarray(item_input), np.asarray(labels))
        print "Begin training..."
        # Training
        print len(user_input)
        batch_len = len(user_input) // batch_size
        for batch in xrange(batch_size):
            _, loss = sess.run(model.optimize,
                               feed_dict={input_user: np.expand_dims(user_input, axis=1)[
                                                      batch * batch_len:(batch + 1) * batch_len],
                                          input_item: np.expand_dims(item_input, axis=1)[
                                                      batch * batch_len:(batch + 1) * batch_len],
                                          output: np.expand_dims(labels, axis=1)[
                                                  batch * batch_len:(batch + 1) * batch_len]})
            if batch % 16 == 0:
                print "epoch %d, batch %d/%d, %.2f" % (epoch, batch, batch_size, loss)
        t2 = time()
        print "Training succeeded!"

        if epoch % verbose == 0:
            print "Evaluating..."
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads,
                                           sess, input_user, input_item)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                # if args.out > 0:
                #     model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))


if __name__ == '__main__':
    main()
