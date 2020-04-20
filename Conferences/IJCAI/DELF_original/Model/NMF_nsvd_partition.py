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
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        self.predict
        self.optimize
        # self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predict(self):
        regularizer = self.regularizer
        embedding_users = tf.get_variable("embedding_users", [self.num_users, self.embedding_size],
                                          regularizer=regularizer)
        embedding_items = tf.get_variable("embedding_items", [self.num_items, self.embedding_size],
                                          regularizer=regularizer)

        # user_embedding: batch_len*embedding_size
        user_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_users, self.input_user), axis=1)
        item_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_items, self.input_item), axis=1)

        # user_ratings: batch_len*num_items
        user_ratings = tf.reduce_sum(tf.gather(self.rating_matrix, self.input_user), axis=1)
        item_ratings = tf.reduce_sum(tf.gather(tf.transpose(self.rating_matrix), self.input_item), axis=1)

        # user_ratings_nonzero: batch_len
        user_ratings_nonzero = tf.cast(tf.reduce_sum(user_ratings, axis=1), tf.float32)
        item_ratings_nonzero = tf.cast(tf.reduce_sum(item_ratings, axis=1), tf.float32)

        # # user_nsvd_embedding: batch_len*embedding_size
        # user_nsvd_embedding = tf.expand_dims(tf.reciprocal(tf.sqrt(user_ratings_nonzero + 1)),
        #                                      axis=1) * tf.matmul(user_ratings, embedding_items)
        # item_nsvd_embedding = tf.expand_dims(tf.reciprocal(tf.sqrt(item_ratings_nonzero + 1)),
        #                                      axis=1) * tf.matmul(item_ratings, embedding_users)

        # GMF_u = user_embedding * item_nsvd_embedding
        # # self.y_u = tf.contrib.slim.fully_connected(GMF_u, 1, tf.identity)
        # GMF_i = item_embedding * user_nsvd_embedding
        # # self.y_i = tf.contrib.slim.fully_connected(GMF_i, 1, tf.identity)
        GMF_ui = user_embedding * item_embedding

        embedding_users_mlp = tf.get_variable("embedding_users_mlp", [self.num_users, self.embedding_size],
                                              regularizer=regularizer)
        embedding_items_mlp = tf.get_variable("embedding_items_mlp", [self.num_items, self.embedding_size],
                                              regularizer=regularizer)
        user_embedding_mlp = tf.reduce_sum(tf.nn.embedding_lookup(embedding_users_mlp, self.input_user), axis=1)
        item_embedding_mlp = tf.reduce_sum(tf.nn.embedding_lookup(embedding_items_mlp, self.input_item), axis=1)
        user_nsvd_embedding_mlp = tf.expand_dims(tf.reciprocal(tf.sqrt(user_ratings_nonzero + 1)),
                                                 axis=1) * tf.matmul(user_ratings, embedding_items_mlp)
        item_nsvd_embedding_mlp = tf.expand_dims(tf.reciprocal(tf.sqrt(item_ratings_nonzero + 1)),
                                                 axis=1) * tf.matmul(item_ratings, embedding_users_mlp)

        merge_embedding = tf.concat([user_embedding_mlp, item_embedding_mlp,
                                     user_nsvd_embedding_mlp, item_nsvd_embedding_mlp],
                                    axis=1, name="merge_embedding")
        x = merge_embedding
        for i in xrange(len(self.layers) - 1):
            x = tf.contrib.slim.fully_connected(x, self.layers[i + 1], weights_regularizer=regularizer)

        x = tf.concat([GMF_ui, x], axis=1, name="concat_embedding")
        # x = tf.contrib.slim.fully_connected(x, 64)
        x = tf.contrib.slim.fully_connected(x, 1, tf.identity)
        return tf.nn.sigmoid(x)

    @define_scope
    def optimize(self):
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict, labels=self.output)
        one_constant = tf.constant(1.0, shape=[1, 1])
        loss = tf.reduce_mean(-self.output * tf.log(self.predict + 1e-10)
                                  - (one_constant - self.output) * tf.log(one_constant - self.predict + 1e-10))
        # reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
        # loss += reg_term
        # 1. / 3 * tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_u, labels=self.output) +
        # 1. / 3 * tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_i, labels=self.output) +
        # 1. / 3 * tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_mlp, labels=self.output))
        optimizer = tf.train.AdamOptimizer()
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        return train_op, tf.reduce_mean(loss)
