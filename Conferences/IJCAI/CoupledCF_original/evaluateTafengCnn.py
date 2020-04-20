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
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives,userAttrMat,itemAttrMat, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _userVecMat
    global _userAttrMat
    global _itemAttrMat
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    #_userVecMat=userVecMat
    _userAttrMat=userAttrMat
    _itemAttrMat=itemAttrMat
    _K = K
    
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
    for idx in range(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    user_attr_input, user_id_input, item_id_input, item_attr_input=[],[],[],[]
    for i in range(len(items)):
        user_attr_input.append(_userAttrMat[u])
        user_id_input.append([u])
        item_id_input.append([items[i]])
        item_attr_input.append(_itemAttrMat[i])
    
    user_attr_input_mat=np.array(user_attr_input)
    user_id_input_mat=np.array(user_id_input)
    item_id_input_mat=np.array(item_id_input)
    item_attr_input_mat=np.array(item_attr_input)
    
    # Get prediction scores
    map_item_score = {}
    item_sub_class = item_attr_input_mat[:, 0]
    item_asset_price = item_attr_input_mat[:, 1:]
    predictions = _model.predict([user_attr_input_mat,item_sub_class,item_asset_price,user_id_input_mat,item_id_input_mat],
                                 batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    
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
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
