import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix

class SpectralCF(object):
    def __init__(self, K, graph, n_users, n_items, emb_dim, lr, batch_size, decay,DIR):
        self.model_name = 'GraphCF with eigen decomposition'
        self.graph = graph
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.K = K
        self.decay = decay

        print("SpectralCF: Computing adjacient_matrix...")
        self.A = self.adjacient_matrix(self_connection=True)

        print("SpectralCF: Computing degree_matrix...")
        self.D = self.degree_matrix()

        print("SpectralCF: Computing laplacian_matrix...")
        self.L = self.laplacian_matrix(normalized=True)

        print("SpectralCF: Computing eigenvalues...")
        self.lamda, self.U = np.linalg.eig(self.L)
        self.lamda = np.diag(self.lamda)

        print("SpectralCF: Building Tensorflow graph...")

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.pos_items = tf.placeholder(tf.int32, shape=(self.batch_size, ))
        self.neg_items = tf.placeholder(tf.int32, shape=(self.batch_size,))


        self.user_embeddings = tf.Variable(
            tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='user_embeddings')
        self.item_embeddings = tf.Variable(
            tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name='item_embeddings')

        self.filters = []
        for k in range(self.K):
            self.filters.append(
                tf.Variable(
                    tf.random_normal([self.emb_dim, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32))

            )


        A_hat = np.dot(self.U, self.U.T) + np.dot(np.dot(self.U, self.lamda), self.U.T)
        #A_hat += np.dot(np.dot(self.U, self.lamda_2), self.U.T)
        A_hat = A_hat.astype(np.float32)

        embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = [embeddings]
        for k in range(0, self.K):

            embeddings = tf.matmul(A_hat, embeddings)

            #filters = self.filters[k]#tf.squeeze(tf.gather(self.filters, k))
            embeddings = tf.nn.sigmoid(tf.matmul(embeddings, self.filters[k]))
            all_embeddings += [embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        self.u_embeddings, self.i_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        self.u_embeddings = tf.nn.embedding_lookup(self.u_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.i_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.i_embeddings, self.neg_items)

        self.all_ratings = tf.matmul(self.u_embeddings, self.i_embeddings, transpose_a=False, transpose_b=True)


        self.loss = self.create_bpr_loss(self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings)


        self.opt = tf.train.RMSPropOptimizer(learning_rate=lr)

        self.updates = self.opt.minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings] + self.filters)

        print("SpectralCF: Building Tensorflow graph... done!")


    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_mean(maxi)) + self.decay * regularizer
        return loss


    def adjacient_matrix(self, self_connection=False):
        A = np.zeros([self.n_users+self.n_items, self.n_users+self.n_items], dtype=np.float32)
        A[:self.n_users, self.n_users:] = self.graph
        A[self.n_users:, :self.n_users] = self.graph.T
        if self_connection == True:
            return np.identity(self.n_users+self.n_items,dtype=np.float32) + A
        return A

    def degree_matrix(self):
        degree = np.sum(self.A, axis=1, keepdims=False)
        #degree = np.diag(degree)
        return degree


    def laplacian_matrix(self, normalized=False):
        if normalized == False:
            return self.D - self.A

        temp = np.dot(np.diag(np.power(self.D, -1)), self.A)
        #temp = np.dot(temp, np.power(self.D, -0.5))
        return np.identity(self.n_users+self.n_items,dtype=np.float32) - temp






