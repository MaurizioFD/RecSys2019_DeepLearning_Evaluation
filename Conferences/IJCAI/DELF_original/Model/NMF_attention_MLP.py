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
        self.batch_len = batch_len
        self.predict
        self.optimize
        # self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predict(self):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        def tf_repeat(tensor, repeats):
            """
            Args:

            input: A Tensor. 1-D or higher.
            repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

            Returns:

            A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
            """
            with tf.variable_scope("repeat"):
                expanded_tensor = tf.expand_dims(tensor, -1)
                multiples = [1] + repeats
                tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
                repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
            return repeated_tesnor

        def attention_self(x, variable_scope):
            with tf.variable_scope(variable_scope):
                W_a0 = tf.get_variable('W_a0', [self.embedding_size, 64])
                b_a0 = tf.get_variable('b_a0', [64, ])
                x = tf.nn.relu(tf.matmul(x, W_a0) + b_a0)
                W_a1 = tf.get_variable('W_a1', [64, 1])
                b_a1 = tf.get_variable('b_a1', [1, ])
                x = tf.matmul(x, W_a1) + b_a1
                return x

        def attention_item_cocat(x, item, variable_scope, func='product'):
            with tf.variable_scope(variable_scope):
                if func == 'concat':
                    x = tf.concat([x, item], axis=1)
                elif func == 'product':
                    x = x * item
                else:
                    raise Exception("Wrong function!")
                W_a0 = tf.get_variable('W_a0', [self.embedding_size, 64])
                b_a0 = tf.get_variable('b_a0', [64, ])
                x = tf.nn.relu(tf.matmul(x, W_a0) + b_a0)
                W_a1 = tf.get_variable('W_a1', [64, 1])
                b_a1 = tf.get_variable('b_a1', [1, ])
                x = tf.matmul(x, W_a1) + b_a1
                return x

        embedding_users = tf.get_variable("embedding_users", [self.num_users, self.embedding_size],
                                          regularizer=regularizer)
        embedding_items = tf.get_variable("embedding_items", [self.num_items, self.embedding_size],
                                          regularizer=regularizer)
        self.embedding_users = embedding_users
        self.embedding_items = embedding_items

        # embedding_users = tf.stop_gradient(embedding_users)
        # embedding_items = tf.stop_gradient(embedding_items)

        batch_len = self.input_user.get_shape().as_list()[0]
        # batch_len = self.batch_len

        # user_embedding: batch_len*embedding_size
        user_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_users, self.input_user), axis=1)
        item_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_items, self.input_item), axis=1)
        # user_embedding = tf.stop_gradient(
        #     tf.reduce_sum(tf.nn.embedding_lookup(embedding_users, self.input_user), axis=1))
        # item_embedding = tf.stop_gradient(
        #     tf.reduce_sum(tf.nn.embedding_lookup(embedding_items, self.input_item), axis=1))

        embedding_items_mlp = tf.get_variable("embedding_items_mlp", [self.num_items, self.embedding_size],
                                              regularizer=regularizer)
        item_embedding_mlp = tf.reduce_sum(tf.nn.embedding_lookup(embedding_items_mlp, self.input_item), axis=1)
        embedding_users_mlp = tf.get_variable("embedding_users_mlp", [self.num_users, self.embedding_size],
                                              regularizer=regularizer)
        user_embedding_mlp = tf.reduce_sum(tf.nn.embedding_lookup(embedding_users_mlp, self.input_user), axis=1)

        # user_ratings: batch_len*num_items
        user_ratings = tf.reduce_sum(tf.gather(self.rating_matrix, self.input_user), axis=1)
        item_ratings = tf.reduce_sum(tf.gather(tf.transpose(self.rating_matrix), self.input_item), axis=1)

        # user_ratings_nonzero = tf.reduce_sum(user_ratings, axis=1, keep_dims=True)/tf.constant(100.)
        # item_ratings_nonzero = tf.reduce_sum(item_ratings, axis=1, keep_dims=True)/tf.constant(7.)

        # attention_self
        # (num_items,)
        attention_type = "self"
        with tf.variable_scope("user_att_on_item"):
            # N = 10
            if attention_type == "self":
                att_embedding_items = tf.reduce_sum(attention_self(embedding_items, "att_embed_items"), axis=1)
            else:
                # (batch_len*num_items, embedding_size)
                item_embedding_rep = tf_repeat(item_embedding, [self.num_items, 1])
                embedding_items_rep = tf.tile(embedding_items, tf.constant([batch_len, 1]))
                att_embedding_items = tf.squeeze(
                    attention_item_cocat(
                        item_embedding_rep,
                        embedding_items_rep,
                        'att_embed_items'))
                att_embedding_items = tf.reshape(att_embedding_items, [batch_len, self.num_items])
            assert att_embedding_items.get_shape().as_list() == [self.num_items]
            # (batch_len, num_items)
            user_score = user_ratings * att_embedding_items
            inf = tf.constant(value=-np.inf, name="numpy_inf")
            # (batch_len, num_items)
            user_mask_inf = tf.where(tf.equal(user_ratings, tf.zeros_like(user_ratings)),
                                     tf.ones_like(user_ratings) * inf, user_ratings) - tf.constant(1.)
            user_score_mask = user_score + user_mask_inf
            fenmu = tf.pow(tf.reduce_sum(tf.exp(user_score_mask), axis=1), 0.5) + tf.constant(1e-10)
            fenmu = tf.expand_dims(fenmu, axis=1)
            user_score_soft = tf.exp(user_score_mask) / fenmu
            # print user_score_soft.get_shape().as_list()
            assert user_score_soft.get_shape().as_list() == [batch_len, self.num_items]
            # (batch_len, embedding_size)
            user_att_embedding = tf.matmul(user_score_soft, embedding_items)
            # user_att_embedding = tf.matmul(user_score_soft, tf.stop_gradient(embedding_items))
            assert user_att_embedding.get_shape().as_list() == [batch_len, self.embedding_size]

        with tf.variable_scope("item_att_on_user"):
            if attention_type == "self":
                att_embedding_users = tf.reduce_sum(attention_self(embedding_users, "att_embed_users"), axis=1)
            else:
                user_embedding_rep = tf_repeat(user_embedding, [self.num_users, 1])
                embedding_users_rep = tf.tile(embedding_users, tf.constant([batch_len, 1]))
                att_embedding_users = tf.squeeze(
                    attention_item_cocat(
                        user_embedding_rep,
                        embedding_users_rep,
                        'att_embed_users'))
                att_embedding_users = tf.reshape(att_embedding_users, [batch_len, self.num_users])

            # (batch_len, num_items)
            item_score = item_ratings * att_embedding_users
            inf = tf.constant(value=-np.inf, name="numpy_inf")
            # (batch_len, num_items)
            item_mask_inf = tf.where(tf.equal(item_ratings, tf.zeros_like(item_ratings)),
                                     tf.ones_like(item_ratings) * inf, item_ratings) - tf.constant(1.)
            item_score_mask = item_score + item_mask_inf
            fenmu_i = tf.pow(tf.reduce_sum(tf.exp(item_score_mask), axis=1), 0.5) + tf.constant(1e-10)
            fenmu_i = tf.expand_dims(fenmu_i, axis=1)
            item_score_soft = tf.exp(item_score_mask) / fenmu_i
            # (batch_len, embedding_size)
            item_att_embedding = tf.matmul(item_score_soft, embedding_users)
            # item_att_embedding = tf.matmul(item_score_soft, tf.stop_gradient(embedding_users))

        # user_ratings_nonzero: batch_len*1
        user_ratings_nonzero = tf.cast(tf.reduce_sum(user_ratings, axis=1, keep_dims=True), tf.float32)
        item_ratings_nonzero = tf.cast(tf.reduce_sum(item_ratings, axis=1, keep_dims=True), tf.float32)

        def mlp(embed_1, embed_2, variable_scope):
            with tf.variable_scope(variable_scope):
                y = tf.concat([embed_1, embed_2], axis=1)
                # y = tf.contrib.layers.fully_connected(y, 128)
                # y = tf.contrib.layers.fully_connected(y, 64)
                y = tf.contrib.layers.fully_connected(y, int(self.layers[1]/2))
                y = tf.contrib.layers.fully_connected(y, int(self.layers[2]/2))
                # y = tf.contrib.layers.fully_connected(y, 1, tf.identity)
                return y

        user_embedding = user_embedding_mlp
        item_embedding = item_embedding_mlp
        #
        ui_y = mlp(user_embedding, item_embedding, "ui_y")
        uni_y = mlp(user_att_embedding, item_embedding, "uni_y")
        uin_y = mlp(user_embedding, item_att_embedding, "uin_y")
        unin_y = mlp(user_att_embedding, item_att_embedding, "unin_y")

        # lambda_user = 2.
        # # alpha_user = tf.clip_by_value((user_ratings_nonzero - 10.) / (lambda_user - 10.), 0., 1.)
        # alpha_user = tf.clip_by_value(tf.sqrt(user_ratings_nonzero) / lambda_user, 0., 1.)
        # lambda_item = 2.
        # # alpha_item = tf.clip_by_value((item_ratings_nonzero - 10.) / (lambda_item - 10.), 0., 1.)
        # alpha_item = tf.clip_by_value(tf.sqrt(item_ratings_nonzero) / lambda_item, 0., 1.)

        # with tf.variable_scope("trainable_alpha"):
        #     W_alpha_u = tf.get_variable("W_alpha_u", shape=(1,1))
        #     b_alpha_u = tf.get_variable("b_alpha_u", shape=(1,1))
        #     alpha_user = tf.sigmoid(user_ratings_nonzero*W_alpha_u+b_alpha_u, "alpha_user")
        #     W_alpha_i = tf.get_variable("W_alpha_i", shape=(1,1))
        #     b_alpha_i = tf.get_variable("b_alpha_i", shape=(1,1))
        #     alpha_item = tf.sigmoid(item_ratings_nonzero*W_alpha_i+b_alpha_i, "alpha_item")
        # self.watch_variables = [W_alpha_u, b_alpha_u, W_alpha_i, b_alpha_i]

        # W_gating = tf.get_variable("W_gating", shape=(1, 64))
        # score_ui = tf.reduce_sum(ui_y * W_gating, axis=1, keep_dims=True)
        # score_uni = tf.reduce_sum(uni_y * W_gating, axis=1, keep_dims=True)
        # score_uin = tf.reduce_sum(uin_y * W_gating, axis=1, keep_dims=True)
        # score_unin = tf.reduce_sum(unin_y * W_gating, axis=1, keep_dims=True)
        # score = tf.concat([score_ui, score_uni, score_uin, score_unin], axis=1)
        # score_soft = tf.nn.softmax(score)
        # x = ui_y * tf.expand_dims(score_soft[:, 0], 1) + uni_y * tf.expand_dims(score_soft[:, 1], 1) + \
        #     uin_y * tf.expand_dims(score_soft[:, 2], 1) + unin_y * tf.expand_dims(score_soft[:, 3], 1)

        x = tf.concat([ui_y, uni_y, uin_y, unin_y], axis=1)
        # x = tf.concat([user_embedding_mlp,user_att_embedding,
        #                item_embedding_mlp,item_att_embedding], axis=1)
        # x = tf.contrib.layers.fully_connected(x, 128)
        # x = tf.contrib.layers.fully_connected(x, 64)

        # alpha_user = 0.5
        # alpha_item = 0.5

        # x = ui_y * alpha_user * alpha_item + uni_y * (1 - alpha_user) * alpha_item + uin_y * alpha_user * (
        #     1 - alpha_item) + unin_y * (1 - alpha_user) * (1 - alpha_item)
        x = tf.contrib.layers.fully_connected(x, self.predictor_size)
        x = tf.contrib.layers.fully_connected(x, 1, tf.identity)
        # x = ui_y * 0.25 + uni_y * 0.25 + uin_y * 0.25 + unin_y * 0.25

        # self.ui_y = ui_y
        # self.uni_y = uni_y
        # self.uin_y = uin_y
        # self.unin_y = unin_y
        # self.alpha_user = alpha_user
        # self.alpha_item = alpha_item
        # self.user_ratings_nonzero = user_ratings_nonzero
        # self.item_ratings_nonzero = item_ratings_nonzero

        print (x.get_shape().as_list())
        # x_u = merge_embedding_u
        # x_i = merge_embedding_i
        # for i in xrange(len(self.layers) - 1):
        #     x_u = tf.contrib.slim.fully_connected(x_u, self.layers[i + 1] // 2)
        #     x_i = tf.contrib.slim.fully_connected(x_i, self.layers[i + 1] // 2)
        # x = tf.concat([x_u, x_i], axis=1)
        # x = tf.contrib.slim.fully_connected(x, 128)
        # x = tf.contrib.slim.fully_connected(x, 64)
        # x = tf.contrib.slim.fully_connected(x, 1, tf.identity)

        # return [x, item_embedding_rep, embedding_items_rep, att_embedding_items]
        return x

    @define_scope
    def optimize(self):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict, labels=self.output)
        # optimizer = tf.train.GradientDescentOptimizer(lr)
        # var_list = [v for v in tf.global_variables() if "trainable_alpha" in v.name]
        optimizer = tf.train.AdamOptimizer()
        # gvs = optimizer.compute_gradients(loss,var_list=var_list)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(grad, var) if grad is None else (tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        return train_op, tf.reduce_mean(loss)

    def optimize_sgd(self, lr):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict, labels=self.output) + tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # optimizer = tf.train.GradientDescentOptimizer(lr)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(grad, var) if grad is None else (tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        return train_op, tf.reduce_mean(loss)
