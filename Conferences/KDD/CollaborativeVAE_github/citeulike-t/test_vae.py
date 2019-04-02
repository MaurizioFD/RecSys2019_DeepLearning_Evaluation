import numpy as np
import tensorflow as tf
import scipy.io
import logging
import sys
import h5py
sys.path.append("..")
from lib.vae import VariationalAutoEncoder
from lib.utils import *

np.random.seed(0)
tf.set_random_seed(0)
init_logging("vae.log")

logging.info('loading data')
# variables = scipy.io.loadmat("../data/citeulike-t/mult_nor.mat")
# data = variables['X']
f = h5py.File('../data/citeulike-t/mult_nor.mat','r') 
data = f.get('X')
data = np.array(data).T
print data.shape

idx = np.random.rand(data.shape[0]) < 0.8
train_X = data[idx]
test_X = data[~idx]
logging.info('initializing sdae model')
model = VariationalAutoEncoder(input_dim=20000, dims=[200, 100], z_dim=50, 
	activations=['sigmoid','sigmoid'], epoch=[50, 50], 
	noise='mask-0.3' ,loss='cross-entropy', lr=0.01, batch_size=128, print_step=1)
logging.info('fitting data starts...')
model.fit(train_X, test_X)
# feat = model.transform(data)
# scipy.io.savemat('feat-dae.mat',{'feat': feat})
# np.savez("sdae-weights.npz", en_weights=model.weights, en_biases=model.biases,
# 	de_weights=model.de_weights, de_biases=model.de_biases)
