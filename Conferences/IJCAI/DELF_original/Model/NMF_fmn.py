import functools
import tensorflow as tf
import numpy as np

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Model:
    def __init__(self, input_user, input_item, output, num_users, num_items, rating_matrix, layers):
        self.input_user = input_user
        self.input_item = input_item
        # self.rating_matrix = tf.constant(rating_matrix.toarray(), dtype=tf.float32)
        self.rating_matrix = rating_matrix
        self.output = output
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = layers[0] // 2
        self.layers = layers
        self.predict
        self.optimize
        print ("Using new model!")
        # self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predict(self):
        embedding_users = tf.get_variable("embedding_users", [self.num_users, self.embedding_size])
        embedding_items = tf.get_variable("embedding_items", [self.num_items, self.embedding_size])
        bias_users = tf.get_variable("bias_users", [self.num_users, 1])
        bias_items = tf.get_variable("bias_items", [self.num_items, 1])

        # user_embedding: batch_len*embedding_size
        # user_bias: batch_len*1
        user_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_users, self.input_user), axis=1)
        item_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_items, self.input_item), axis=1)
        user_bias = tf.reduce_sum(tf.nn.embedding_lookup(bias_users, self.input_user), axis=1)
        item_bias = tf.reduce_sum(tf.nn.embedding_lookup(bias_items, self.input_item), axis=1)

        # user_ratings: batch_len*num_items
        user_ratings = tf.reduce_sum(tf.gather(self.rating_matrix, self.input_user), axis=1)
        r_minus_b = (user_ratings - (user_bias + tf.transpose(bias_items))) * user_ratings  # batch_len*num_items

        wij = tf.matmul(item_embedding, tf.transpose(embedding_items))  # batch_len*num_items
        item_embedding_norm = tf.norm(item_embedding, ord=2, axis=1, keep_dims=True)
        embed_items_norm = tf.norm(tf.transpose(embedding_items), ord=2, axis=0, keep_dims=True)
        wij = wij * tf.reciprocal(tf.matmul(item_embedding_norm, embed_items_norm)) * user_ratings
        # wij_reduce = tf.expand_dims(tf.reduce_sum(wij,axis=1),axis=1)
        wij_reduce = tf.expand_dims(tf.sqrt(tf.reduce_sum(user_ratings, axis=1)),axis=1)
        wij = wij/wij_reduce
        # wij = tf.where(tf.equal(wij, 0), tf.zeros_like(wij) - np.inf, wij)
        # print "shape of wij"
        # print wij.get_shape().as_list()
        # wij = tf.nn.softmax(wij,1)
        neighbor_rating = tf.expand_dims(tf.reduce_sum(r_minus_b * wij, axis=1), axis=1) + user_bias + item_bias

        # user_ratings_nonzero: batch_len
        # user_ratings_nonzero = tf.cast(tf.reduce_sum(user_ratings, axis=1), tf.float32)
        # item_ratings_nonzero = tf.cast(tf.reduce_sum(item_ratings, axis=1), tf.float32)
        #
        # user_nsvd_embedding: batch_len*embedding_size
        # user_nsvd_embedding = tf.expand_dims(tf.reciprocal(tf.sqrt(user_ratings_nonzero+1)),
        #                                      axis=1) * tf.matmul(user_ratings, embedding_items)
        # item_nsvd_embedding = tf.expand_dims(tf.reciprocal(tf.sqrt(item_ratings_nonzero+1)),
        #                                      axis=1) * tf.matmul(item_ratings, embedding_users)
        merge_embedding = tf.concat([user_embedding, item_embedding],
                                    axis=1, name="merge_embedding")
        print "merge_embedding shape is:"
        print merge_embedding.get_shape().as_list()
        x = merge_embedding
        for i in xrange(len(self.layers) - 1):
            x = tf.contrib.slim.fully_connected(x, self.layers[i + 1])
        x = tf.contrib.slim.fully_connected(x, 1, tf.sigmoid)
        merge_rating = tf.concat([x, neighbor_rating], axis=1, name="merge_rating")
        x = tf.contrib.slim.fully_connected(merge_rating, 1, tf.identity)
        return x

    @define_scope
    def optimize(self):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict, labels=self.output)
        optimizer = tf.train.AdamOptimizer()
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        return train_op, tf.reduce_mean(loss)
