import functools
import tensorflow as tf


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
    def __init__(self, input_user, input_item, output, num_users, num_items, embedding_size, rating_matrix):
        self.input_user = input_user
        self.input_item = input_item
        self.rating_matrix = tf.constant(rating_matrix.toarray(), dtype=tf.float32)
        self.output = output
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.embedding_users = tf.get_variable("embedding_users", [self.num_users, self.embedding_size])
        self.embedding_items = tf.get_variable("embedding_items", [self.num_items, self.embedding_size])
        self.predict
        self.optimize
        # self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predict(self):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        embedding_users = tf.stop_gradient(self.embedding_users, "embedding_users_nograd")
        embedding_items = tf.stop_gradient(self.embedding_items, "embedding_items_nograd")
        user_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_users, self.input_user), axis=1)
        item_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_items, self.input_item), axis=1)

        user_ratings = tf.reduce_sum(tf.gather(self.rating_matrix, self.input_user), axis=1)
        item_ratings = tf.reduce_sum(tf.gather(tf.transpose(self.rating_matrix), self.input_item), axis=1)
        user_ratings_nonzero = tf.cast(tf.reduce_sum(user_ratings, axis=1), tf.float32)
        item_ratings_nonzero = tf.cast(tf.reduce_sum(item_ratings, axis=1), tf.float32)
        user_nsvd_embedding = tf.expand_dims(tf.reciprocal(user_ratings_nonzero),
                                             axis=1) * tf.matmul(user_ratings, embedding_items)
        item_nsvd_embedding = tf.expand_dims(tf.reciprocal(item_ratings_nonzero),
                                             axis=1) * tf.matmul(item_ratings, embedding_users)
        # merge_embedding = tf.concat([user_embedding, item_embedding,
        #                              tf.stop_gradient(user_nsvd_embedding),
        #                              tf.stop_gradient(item_nsvd_embedding),],
        #                             axis=1, name="merge_embedding")
        merge_embedding = tf.concat([user_embedding, item_embedding,
                                     user_nsvd_embedding,
                                     item_nsvd_embedding, ],
                                    axis=1, name="merge_embedding")
        print "merge_embedding shape is:"
        print merge_embedding.get_shape().as_list()
        x = tf.contrib.slim.fully_connected(merge_embedding, 32, weights_regularizer=regularizer)
        x = tf.contrib.slim.fully_connected(x, 16)
        x = tf.contrib.slim.fully_connected(x, 8)
        x = tf.contrib.slim.fully_connected(x, 1, tf.identity)
        return x

    @define_scope
    def optimize(self):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict, labels=self.output)
        loss = cross_entropy + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict, labels=self.output)
        # loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # optimizer = tf.train.GradientDescentOptimizer(0.01)
        optimizer = tf.train.AdamOptimizer()
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(grad, var) if grad is None else (tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        return train_op, tf.reduce_mean(cross_entropy)
