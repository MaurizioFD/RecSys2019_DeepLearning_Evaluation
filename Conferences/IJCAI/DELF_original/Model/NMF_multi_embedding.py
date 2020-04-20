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
    def __init__(self, input_user, input_item, output, num_users, num_items, layers, MLP=False):
        self.input_user = input_user
        self.input_item = input_item
        self.output = output
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = layers[0] // 2
        self.layers = layers
        self.MLP = MLP
        self.embedding_users = []
        self.embedding_items = []
        self.num_embed = 5
        for i in range(self.num_embed):
            self.embedding_users.append(tf.get_variable("embedding_users_%d" % i, [self.num_users, self.embedding_size]))
            self.embedding_items.append(tf.get_variable("embedding_items_%d" % i, [self.num_items, self.embedding_size]))
        self.predict
        self.optimize
        # self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predict(self):
        with tf.name_scope("embedding"):
            user_embeddings=[]
            item_embeddings=[]
            for i in range(self.num_embed):
                user_embeddings.append(tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_users[i], self.input_user), axis=1))
                item_embeddings.append(tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_items[i], self.input_item), axis=1))
            inter=[]
            for i in range(self.num_embed):
                for j in range(self.num_embed):
                    inter.append(tf.reduce_sum(user_embeddings[i] * item_embeddings[j], axis=1, keep_dims=True))
            inter = tf.concat(inter, axis=1)
            inter = tf.reduce_max(inter, axis=1, keep_dims=True)
            # tf.summary.histogram("user_embedding", user_embedding)
        # x = merge_embedding
        # with tf.name_scope("fc"):
        #     for i in xrange(len(self.layers) - 1):
        #         x = tf.contrib.slim.fully_connected(x, self.layers[i + 1])
        #     if not self.MLP:
        #         x = tf.concat([x, GMF_embed], axis=1, name = "concat_embedding")
        #         print ("MLP is False, Running NeuMF")
        #     x = tf.contrib.slim.fully_connected(x, 1, tf.identity)
        return inter, user_embeddings, item_embeddings

    @define_scope
    def optimize(self):
        with tf.name_scope("optimize"):
            cross_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict[0],
                                                                 labels=self.output, name="cross_entropy")
            # tf.summary.scalar("cross_entropy", loss)
            sim_loss = tf.reduce_sum(self.predict[1][0] * self.predict[1][1], axis=1) / (
            tf.norm(self.predict[1][0], axis=1) * tf.norm(self.predict[1][1], axis=1))
            loss = tf.reduce_sum(cross_loss, 1) + sim_loss
            optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(cross_loss), tf.reduce_mean(loss), tf.reduce_mean(cross_loss), tf.reduce_mean(sim_loss)
