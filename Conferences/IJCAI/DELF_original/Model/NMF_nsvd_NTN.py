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
    def __init__(self, input_user, input_item, output, num_users, num_items, rating_matrix,layers):
        self.input_user = input_user
        self.input_item = input_item
        # self.rating_matrix = tf.constant(rating_matrix.toarray(), dtype=tf.float32)
        self.rating_matrix = rating_matrix
        self.output = output
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = layers[0]//2
        self.layers = layers
        self.predict
        self.optimize
        # self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def predict(self):
        embedding_users = tf.get_variable("embedding_users", [self.num_users, self.embedding_size])
        embedding_items = tf.get_variable("embedding_items", [self.num_items, self.embedding_size])
        self.embedding_users = embedding_users
        self.embedding_items = embedding_items

        #user_embedding: batch_len*embedding_size
        user_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_users, self.input_user), axis=1)
        item_embedding = tf.reduce_sum(tf.nn.embedding_lookup(embedding_items, self.input_item), axis=1)

        #user_ratings: batch_len*num_items
        user_ratings = tf.reduce_sum(tf.gather(self.rating_matrix, self.input_user), axis=1)
        item_ratings = tf.reduce_sum(tf.gather(tf.transpose(self.rating_matrix), self.input_item), axis=1)

        #user_ratings_nonzero: batch_len
        user_ratings_nonzero = tf.cast(tf.reduce_sum(user_ratings, axis=1), tf.float32)
        item_ratings_nonzero = tf.cast(tf.reduce_sum(item_ratings, axis=1), tf.float32)

        #user_nsvd_embedding: batch_len*embedding_size
        # user_nsvd_embedding = tf.expand_dims(tf.reciprocal(user_ratings_nonzero),
        #                                      axis=1) * tf.matmul(user_ratings, embedding_items)
        # item_nsvd_embedding = tf.expand_dims(tf.reciprocal(item_ratings_nonzero),
        #                                      axis=1) * tf.matmul(item_ratings, embedding_users)
        user_nsvd_embedding = tf.expand_dims(tf.reciprocal(tf.sqrt(user_ratings_nonzero+1)),
                                             axis=1) * tf.matmul(user_ratings, embedding_items)
        item_nsvd_embedding = tf.expand_dims(tf.reciprocal(tf.sqrt(item_ratings_nonzero+1)),
                                             axis=1) * tf.matmul(item_ratings, embedding_users)
        def hbijk(embed_a, embed_b, num_slice, variable_scope):
            assert num_slice>=1
            with tf.variable_scope(variable_scope):
                B = tf.get_variable("B_0", [self.embedding_size, self.embedding_size])
                hb = tf.expand_dims(tf.reduce_sum(tf.matmul(embed_a,B)*embed_b,axis=1),axis=1)
                for i in xrange(num_slice-1):
                    B = tf.get_variable("B_%d"%(i+1), [self.embedding_size, self.embedding_size])
                    hb_this = tf.expand_dims(tf.reduce_sum(tf.matmul(embed_a,B)*embed_b,axis=1),axis=1)
                    hb = tf.concat([hb, hb_this], axis = 1)
            return hb
        num_slice = 4
        hb_u0i0 = hbijk(user_embedding, item_embedding, num_slice, "u0i0")
        hb_u0i1 = hbijk(user_embedding, item_nsvd_embedding, num_slice, "u0i1")
        hb_u1i0 = hbijk(user_nsvd_embedding, item_embedding, num_slice, "u1i0")
        hb_u1i1 = hbijk(user_nsvd_embedding, item_nsvd_embedding, num_slice, "u1i1")

        # merge_embedding = tf.concat([user_embedding, item_embedding,
        #                              tf.stop_gradient(user_nsvd_embedding),
        #                              tf.stop_gradient(item_nsvd_embedding),],
        #                             axis=1, name="merge_embedding")
        merge_embedding = tf.concat([user_embedding, item_embedding,
                                     user_nsvd_embedding,
                                     item_nsvd_embedding, ],
                                    axis=1, name="merge_embedding")
        ha = tf.contrib.slim.fully_connected(merge_embedding, 32, tf.identity)
        f = tf.tanh(tf.concat([ha, hb_u0i0, hb_u0i1, hb_u1i0, hb_u1i1], axis=1),name="g_ha_hb")
        print "g(ha,hb) shape is: (should be none*48)"
        print f.get_shape().as_list()
        f = tf.contrib.slim.fully_connected(f, 1, tf.identity)
        # print "merge_embedding shape is:"
        # print merge_embedding.get_shape().as_list()
        # x=merge_embedding
        # for i in xrange(len(self.layers)-1):
        #     x = tf.contrib.slim.fully_connected(x,self.layers[i+1] )
        # x = tf.contrib.slim.fully_connected(x, 1, tf.identity)
        return f

    @define_scope
    def optimize(self):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict, labels=self.output)
        optimizer = tf.train.AdamOptimizer()
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        return train_op, tf.reduce_mean(loss)
