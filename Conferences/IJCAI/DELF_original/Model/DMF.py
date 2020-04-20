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
    def __init__(self, input_user, input_item, output, num_users, num_items, rating_matrix, layers, batch_len):
        self.input_user = input_user
        self.input_item = input_item
        # self.rating_matrix = tf.constant(rating_matrix.toarray(), dtype=tf.float32)
        self.rating_matrix = rating_matrix
        self.output = output
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = layers[0] // 2
        self.predictor_size = layers[-1]
        self.layers = layers
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        self.batch_len = batch_len
        self.predict
        self.optimize
        # self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predict(self):
        regularizer = self.regularizer
        user_ratings = tf.reduce_sum(tf.gather(self.rating_matrix, self.input_user), axis=1)
        item_ratings = tf.reduce_sum(tf.gather(tf.transpose(self.rating_matrix), self.input_item), axis=1)
        print("shape of u_r:")
        print user_ratings.get_shape().as_list()
        print("shape of i_r:")
        print item_ratings.get_shape().as_list()
        u_x = tf.contrib.layers.fully_connected(user_ratings, 300,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.contrib.layers.xavier_initializer())
        u_x = tf.contrib.layers.fully_connected(u_x, self.predictor_size, tf.identity,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.contrib.layers.xavier_initializer())
        i_x = tf.contrib.layers.fully_connected(item_ratings, 300,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.contrib.layers.xavier_initializer())
        i_x = tf.contrib.layers.fully_connected(i_x, self.predictor_size, tf.identity,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.contrib.layers.xavier_initializer())
        # weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        fen_zi = tf.reduce_sum(u_x * i_x, 1, keep_dims=True)

        norm_u = tf.sqrt(tf.reduce_sum(tf.square(u_x), 1, keep_dims=True))
        norm_v = tf.sqrt(tf.reduce_sum(tf.square(i_x), 1, keep_dims=True))
        fen_mu = norm_u * norm_v

        y = tf.nn.relu(fen_zi / fen_mu)
        return y

    @define_scope
    def optimize(self):
        one_constant = tf.constant(1.0, shape=[1, 1])
        loss = tf.reduce_mean(-self.output * tf.log(self.predict + 1e-10)
                                  - (one_constant - self.output) * tf.log(one_constant - self.predict + 1e-10))
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict, labels=self.output)
        # reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
        # loss += reg_term
        # 1. / 3 * tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_u, labels=self.output) +
        # 1. / 3 * tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_i, labels=self.output) +
        # 1. / 3 * tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_mlp, labels=self.output))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -0.01, 0.01), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        return train_op, tf.reduce_mean(loss)
