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
_rating_matrix = None
_train = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread, sess, input_user, input_item, rating_matrix,train):
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
    global _rating_matrix
    global _train

    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    _sess = sess
    _input_user = input_user
    _input_item = input_item
    _rating_matrix = rating_matrix
    _train = train
    batch_size=512
        
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
    for idx in xrange(len(_testRatings)):
        if idx%(len(_testRatings)//100)==0:
            print ("%d/100 done...,[%.1f s]")%(idx/(len(_testRatings)//100),time()-time_1)
            time_1=time()
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    sess = _sess
    input_user=_input_user
    input_item=_input_item
    rating_matrix=_rating_matrix
    train= _train

    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')

    predictions = sess.run(_model.predict,
                           feed_dict={input_user:np.expand_dims(users, axis=1),
                                      input_item:np.expand_dims(np.array(items), axis=1),
                                      rating_matrix: train})
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
