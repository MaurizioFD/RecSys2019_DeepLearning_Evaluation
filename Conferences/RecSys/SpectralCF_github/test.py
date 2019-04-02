import Conferences.RecSys.SpectralCF_github.utils as ut
from Conferences.RecSys.SpectralCF_github.params import *
from Conferences.RecSys.SpectralCF_github.load_data import *
import multiprocessing
cores = multiprocessing.cpu_count()

data_generator = Data(train_file=DIR+'train_users.dat', test_file=DIR+'test_users.dat',batch_size=BATCH_SIZE)
USER_NUM, ITEM_NUM = data_generator.get_num_users_items()


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    training_items = data_generator.train_items[u]
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    recall_20 = ut.recall_at_k(r, 20, len(user_pos_test))
    recall_40 = ut.recall_at_k(r, 40, len(user_pos_test))
    recall_60 = ut.recall_at_k(r, 60, len(user_pos_test))
    recall_80 = ut.recall_at_k(r, 80, len(user_pos_test))
    recall_100 = ut.recall_at_k(r, 100, len(user_pos_test))

    ap_20 = ut.average_precision(r,20)
    ap_40 = ut.average_precision(r, 40)
    ap_60 = ut.average_precision(r, 60)
    ap_80 = ut.average_precision(r, 80)
    ap_100 = ut.average_precision(r, 100)


    return np.array([recall_20,recall_40,recall_60,recall_80,recall_100, ap_20,ap_40,ap_60,ap_80,ap_100])


def test(sess, model, users_to_test):
    result = np.array([0.] * 10)
    pool = multiprocessing.Pool(cores)
    batch_size = BATCH_SIZE
    #all users needed to test
    test_users = users_to_test
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size
        FLAG = False
        if len(user_batch) < batch_size:
            user_batch += [user_batch[-1]] * (batch_size - len(user_batch))
            user_batch_len = len(user_batch)
            FLAG = True
        user_batch_rating = sess.run(model.all_ratings, {model.users: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        if FLAG == True:
            batch_result = batch_result[:user_batch_len]
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret