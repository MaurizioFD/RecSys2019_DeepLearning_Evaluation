from Conferences.RecSys.SpectralCF_github.SpectralCF import SpectralCF
from Conferences.RecSys.SpectralCF_github.test import *

import tensorflow as tf

def main():

    print("Instantiating model...")

    model = SpectralCF(K=K, graph=data_generator.R, n_users=USER_NUM, n_items=ITEM_NUM, emb_dim=EMB_DIM,
                     lr=LR, decay=DECAY, batch_size=BATCH_SIZE,DIR=DIR)

    print("Instantiating model... done!")
    print(model.model_name)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print("Training model... ")

    for epoch in range(N_EPOCH):
        users, pos_items, neg_items = data_generator.sample()
        _, loss = sess.run([model.updates, model.loss],
                                           feed_dict={model.users: users, model.pos_items: pos_items,
                                                      model.neg_items: neg_items})

        users_to_test = list(data_generator.test_set.keys())

        ret = test(sess, model, users_to_test)


        print('Epoch %d training loss %f' % (epoch, loss))
        print('recall_20 %f recall_40 %f recall_60 %f recall_80 %f recall_100 %f'
              % (ret[0],ret[1],ret[2],ret[3],ret[4]))
        print('map_20 %f map_40 %f map_60 %f map_80 %f map_100 %f'
              % (ret[5], ret[6], ret[7], ret[8], ret[9]))

    print("Training model... done!")


if __name__ == '__main__':
    main()
