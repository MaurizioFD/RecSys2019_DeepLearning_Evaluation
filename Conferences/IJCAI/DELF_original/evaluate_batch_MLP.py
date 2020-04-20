'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
import tensorflow as tf

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

_sess = None
_input_user = None
_input_item = None
# _rating_matrix = None
# _train = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread, sess, input_user, input_item):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K

    global _sess
    global _input_user
    global _input_item
    # global _rating_matrix
    # global _train

    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    _sess = sess
    _input_user = input_user
    _input_item = input_item
    # _rating_matrix = rating_matrix
    # _train = train
    # batch_size=256
    # batch_len = len(_testRatings)//batch_size
    # batch_len_last = len(_testRatings) % batch_size
    batch_len = 50
    batch_size = (len(_testRatings)-1) // batch_len
    # batch_len_last = len(_testRatings) % batch_len
    batch_len_last = len(_testRatings) - batch_size*batch_len

    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    time_1 = time()
    for idx in xrange(batch_size+1):
        # if idx%(batch_size//100)==0:
        #     print ("%d/100 done...,[%.1f s]")%(idx/(batch_size//100),time()-time_1)
        #     time_1=time()
        if idx%16==0:
            print ("%d/%d done...,[%.1f s]") % (idx, batch_size, time() - time_1)
        (hr,ndcg) = eval_batch_rating(idx, batch_len, batch_size, batch_len_last)
        hits.extend(hr)
        ndcgs.extend(ndcg)
    return (hits, ndcgs)


def eval_batch_rating(idx, batch_len, batch_size, batch_len_last):
    if idx<batch_size:
        ratings = _testRatings[idx*batch_len:(idx+1)*batch_len] #(batch_len,2)
        items = _testNegatives[idx*batch_len:(idx+1)*batch_len] #(batch_len,len_items)
    elif idx==batch_size:
        ratings = _testRatings[idx*batch_len:] #(batch_len,2)
        items = _testNegatives[idx*batch_len:] #(batch_len,len_items)
        batch_len = batch_len_last

    len_items = len(items[0])
    sess = _sess
    input_user = _input_user
    input_item = _input_item
    # rating_matrix = _rating_matrix
    # train = _train

    users = np.expand_dims(np.array(ratings)[:,0],axis=1) #(batch_len,)
    users_rep = np.repeat(users,len_items+1,axis=1) #(batch_len,len_items+1)
    users_rep = users_rep.flatten() #(batch_len*(len_items+1),)

    gtItem = np.array(ratings)[:,1] #(batch_len,)
    items = np.append(np.array(items),np.expand_dims(gtItem,axis=1),axis=1)#(batch_len,len_items+1)
    # print "shape of input_user"

    # print users_rep.shape
    # Get prediction scores
    predictions = sess.run(_model.predict,
                           feed_dict={input_user: np.expand_dims(users_rep, axis=1),#(batch_len*(len_items+1),1)
                                      input_item: np.expand_dims(items.flatten(), axis=1)})[0]
    #predictions: (batch_len*(len_items+1),1) a = np.random.randint(0, 20, (10, 10))
    # print "Shape of predictions:"
    # print predictions.shape

    predictions = np.reshape(predictions,[batch_len,len_items+1])
    predictions_topk = np.argsort(predictions, axis=1)[:, -_K:]
    # predictions_topk = np.argpartition(predictions, np.argmin(predictions, axis=1))[:, -_K:]
    predictions_topk = np.flip(predictions_topk,axis=1) #batch_len*_K
    row_index = np.repeat(np.expand_dims(np.arange(batch_len),axis=1),_K,axis=1).flatten()
    column_index = predictions_topk.flatten()
    rank_array = items[row_index, column_index].reshape((batch_len,_K))

    hr = getHitRatio_batch(rank_array, gtItem)
    ndcg = getNDCG_batch(rank_array, gtItem)
    return (hr, ndcg)

def getHitRatio_batch(rank_array, gtItem):
    rank_array = rank_array-np.expand_dims(gtItem,1)
    rank_array_zero = np.equal(rank_array,0.)
    hits = np.sum(rank_array_zero,axis=1) #(batch_len,)
    return hits.tolist()

def getNDCG_batch(rank_array, gtItem):
    rank_array = rank_array-np.expand_dims(gtItem,1)
    rank_array_zero = np.equal(rank_array,0.) #batch_len*_K
    vectors = np.zeros(shape=(rank_array.shape[0],)) #batch_len
    for i,vector in enumerate(rank_array_zero):
        if np.sum(vector)==0:
            vectors[i]=np.inf
        elif np.sum(vector)==1:
            vectors[i]=np.where(vector==True)[0][0]
    vectors = np.log(2)/np.log(vectors+2)
    return vectors.tolist()


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    sess = _sess
    input_user=_input_user
    input_item=_input_item
    # rating_matrix=_rating_matrix
    # train= _train

    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')

    predictions = sess.run(_model.predict,
                           feed_dict={input_user:np.expand_dims(users, axis=1),
                                      input_item:np.expand_dims(np.array(items), axis=1)})
    for i in xrange(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
