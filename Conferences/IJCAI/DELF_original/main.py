import functools
import os
import tensorflow as tf
import argparse
import numpy as np
from evaluate_batch_MLP import evaluate_model
from Dataset import Dataset
from time import time
import Model.NMF_multi_embedding
import Model.MF
import Model.NMF
import atexit

# import Model.MF as NMF
import sys


# def doublewrap(function):
#     """
#     A decorator decorator, allowing to use the decorator to be used without
#     parentheses if not arguments are provided. All arguments must be optional.
#     """
#
#     @functools.wraps(function)
#     def decorator(*args, **kwargs):
#         if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
#             return function(args[0])
#         else:
#             return lambda wrapee: function(wrapee, *args, **kwargs)
#
#     return decorator
#
#
# @doublewrap
# def define_scope(function, scope=None, *args, **kwargs):
#     """
#     A decorator for functions that define TensorFlow operations. The wrapped
#     function will only be executed once. Subsequent calls to it will directly
#     return the result so that operations are added to the graph only once.
#     The operations added by the function live within a tf.variable_scope(). If
#     this decorator is used with arguments, they will be forwarded to the
#     variable scope. The scope name defaults to the name of the wrapped
#     function.
#     """
#     attribute = '_cache_' + function.__name__
#     name = scope or function.__name__
#
#     @property
#     @functools.wraps(function)
#     def decorator(self):
#         if not hasattr(self, attribute):
#             with tf.variable_scope(name, *args, **kwargs):
#                 setattr(self, attribute, function(self))
#         return getattr(self, attribute)
#
#     return decorator
#
#
# class NMF_Model:
#     def __init__(self, input_user, input_item, output, num_users, num_items, embedding_size):
#         self.input_user = input_user
#         self.input_item = input_item
#         self.output = output
#         self.num_users = num_users
#         self.num_items = num_items
#         self.embedding_size = embedding_size
#         self.predict
#         self.optimize
#         # self.error
#
#     @define_scope(initializer=tf.contrib.slim.xavier_initializer())
#     def predict(self):
#         embedding_users = tf.get_variable("embedding_users", [self.num_users, self.embedding_size])
#         embedding_items = tf.get_variable("embedding_items", [self.num_items, self.embedding_size])
#         user_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_users, self.input_user), axis=1)
#         item_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_items, self.input_item), axis=1)
#         merge_embedding = tf.concat([user_embedding, item_embedding], axis=1, name="merge_embedding")
#         # print "merge_embedding shape is:"
#         # print merge_embedding.get_shape().as_list()
#         x = tf.contrib.slim.fully_connected(merge_embedding, 32)
#         x = tf.contrib.slim.fully_connected(x, 16)
#         x = tf.contrib.slim.fully_connected(x, 8)
#         x = tf.contrib.slim.fully_connected(x, 1, tf.identity)
#         return x
#
#     @define_scope
#     def optimize(self):
#         loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict, labels=self.output)
#         optimizer = tf.train.AdamOptimizer()
#         return optimizer.minimize(loss)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--model', nargs='?', default='MF',
                        help='Model to use.')
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='movielens1m-paper',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[256,128,64]',
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
    parser.add_argument('--MLP', type=int, default=1,
                        help='Whether to apply only the MLP model.')
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
    topK = 100
    evaluation_threads = 1  # mp.cpu_count()
    predictive_factors = eval(args.layers)[-1]
    MLP = bool(args.MLP)
    print("MLP arguments: %s " % (args))
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def exit_handler():
        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
        print("Test performance in Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (
            best_iter, best_hr_test_byvalid, best_ndcg_test_byvalid))
        print("Best test Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter_test, best_hr_test, best_ndcg_test))

        np.save("output/valid_ndcgs_sgd.npy", np.asarray(valid_ndcgs_collection))
        np.save("output/test_ndcgs_sgd.npy",np.asarray(test_ndcgs_collection))

        np.save("output/user_scale.npy", np.asarray(user_scale_collection))
        np.save("output/item_scale.npy", np.asarray(item_scale_collection))

    atexit.register(exit_handler)

    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, validRatings, testRatings, testNegatives = dataset.trainMatrix, dataset.validRatings, \
                                                      dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    input_user = tf.placeholder(tf.int32, [None, 1], name="input_user")
    input_item = tf.placeholder(tf.int32, [None, 1], name="input_item")
    output = tf.placeholder(tf.float32, [None, 1], name="output")

    # args.model = "MF"
    NMFs = {"multi-embed": Model.NMF_multi_embedding, 'MF': Model.MF,
            "NMF": Model.NMF}
    NMF = NMFs[args.model]

    model = NMF.Model(input_user, input_item, output, num_users, num_items, layers, MLP=MLP)
    tf.summary.histogram("input_user", input_user)
    merged_summary = tf.summary.merge_all()

    # saver_model = tf.train.Saver()
    # saver = tf.train.Saver({"embedding_users": model.embedding_users,
    #                         "embedding_items": model.embedding_items})

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.initialize_all_variables())
    # saver.restore(sess, "/Dataset/weiyu/Pretrain/MLP_%s_embeddings"%args.dataset)
    writer = tf.summary.FileWriter("./tmp/NMF")
    writer.add_graph(sess.graph)

    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads,
                                   sess, input_user, input_item)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    best_hr_test, best_ndcg_test = -np.inf, -np.inf

    valid_ndcgs_collection = []
    test_ndcgs_collection = []
    user_scale_collection = []
    item_scale_collection = []

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
            _, loss, c_loss, s_loss = sess.run(model.optimize,
                                               feed_dict={input_user: np.expand_dims(user_input, axis=1)[
                                                                      batch * batch_len:(batch + 1) * batch_len],
                                                          input_item: np.expand_dims(item_input, axis=1)[
                                                                      batch * batch_len:(batch + 1) * batch_len],
                                                          output: np.expand_dims(labels, axis=1)[
                                                                  batch * batch_len:(batch + 1) * batch_len]})
            if batch  == 0 and args.model=="multi-embed":
                print "epoch %d, batch %d/%d, loss: %.3f, c_loss: %.3f, s_loss: %.3f" % (
                epoch, batch, batch_size, loss, c_loss, s_loss)
            # _, loss = sess.run(model.optimize,
            #                    feed_dict={input_user: np.expand_dims(user_input, axis=1)[
            #                                           batch * batch_len:(batch + 1) * batch_len],
            #                               input_item: np.expand_dims(item_input, axis=1)[
            #                                           batch * batch_len:(batch + 1) * batch_len],
            #                               output: np.expand_dims(labels, axis=1)[
            #                                       batch * batch_len:(batch + 1) * batch_len]})
            # if batch % 16 == 0:
            #     print "epoch %d, batch %d/%d, loss: %.3f" % (
            #         epoch, batch, batch_size, loss)
            if batch == 0:
                s = sess.run(merged_summary,
                             feed_dict={input_user: np.expand_dims(user_input, axis=1)[
                                                    batch * batch_len:(batch + 1) * batch_len],
                                        input_item: np.expand_dims(item_input, axis=1)[
                                                    batch * batch_len:(batch + 1) * batch_len],
                                        output: np.expand_dims(labels, axis=1)[
                                                batch * batch_len:(batch + 1) * batch_len]})
                writer.add_summary(s, epoch)
        t2 = time()
        print "Training succeeded!"
        user_scale, item_scale = sess.run(model.status)
        user_scale_collection.append(user_scale)
        item_scale_collection.append(item_scale)

        if epoch % verbose == 0:
            print "Evaluating on valid set..."
            (hits, ndcgs) = evaluate_model(model, validRatings, testNegatives, topK, evaluation_threads,
                                           sess, input_user, input_item)
            valid_ndcgs_collection.append(ndcgs)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, time() - t2))

            print "Evaluating on test set..."
            t2 = time()
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads,
                                           sess, input_user, input_item)
            test_ndcgs_collection.append(ndcgs)
            hr_test, ndcg_test = np.array(hits).mean(), np.array(ndcgs).mean()
            if (hr_test + ndcg_test) / 2. > (best_hr_test + best_ndcg_test) / 2.:
                best_hr_test, best_ndcg_test = hr_test, ndcg_test
                best_iter_test = epoch

            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]'
                  % (epoch, t2 - t1, hr_test, ndcg_test, time() - t2))
            if (hr + ndcg) / 2. > (best_hr + best_ndcg) / 2.:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                best_hr_test_byvalid, best_ndcg_test_byvalid = hr_test, ndcg_test
                # if args.out > 0:
                #     if MLP:
                #         # save_path = saver.save(sess, "/Dataset/weiyu/Pretrain/MLP_%s_embeddings_%s" % (
                #         #     args.dataset, predictive_factors))
                #         save_path = saver.save(sess, "Pretrain/MLP_%s_embeddings_%s" % (
                #             args.dataset, predictive_factors))
                #         print("Model saved in file: %s" % save_path)
    np.save("output/user_scale.npy", np.asarray(user_scale_collection))
    np.save("output/item_scale.npy", np.asarray(item_scale_collection))

    np.save("output/valid_ndcgs_sgd.npy", np.asarray(valid_ndcgs_collection))
    np.save("output/test_ndcgs_sgd.npy", np.asarray(test_ndcgs_collection))
    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    print("Test performance in Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (
        best_iter, best_hr_test_byvalid, best_ndcg_test_byvalid))
    print("Best test Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter_test, best_hr_test, best_ndcg_test))


if __name__ == '__main__':
    main()
