from Conferences.KDD.CollaborativeVAE_github.lib.cvae import *
import numpy as np
import tensorflow as tf
import scipy.io
from Conferences.KDD.CollaborativeVAE_github.lib.utils import *

np.random.seed(0)
tf.set_random_seed(0)
init_logging("cvae.log")

def load_cvae_data():
  data = {}
  data_dir = "./data/citeulike-a/"
  variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
  data["content"] = variables['X']

  data["train_users"] = load_rating(data_dir + "cf-train-1-users.dat")
  data["train_items"] = load_rating(data_dir + "cf-train-1-items.dat")
  data["test_users"] = load_rating(data_dir + "cf-test-1-users.dat")
  data["test_items"] = load_rating(data_dir + "cf-test-1-items.dat")

  return data

def load_rating(path):
  arr = []
  for line in open(path):
    a = line.strip().split()
    if a[0]==0:
      l = []
    else:
      l = [int(x) for x in a[1:]]
    arr.append(l)
  return arr

params = Params()
params.lambda_u = 0.1
params.lambda_v = 10
params.lambda_r = 1
params.a = 1
params.b = 0.01
params.M = 300
params.n_epochs = 100
params.max_iter = 1

data = load_cvae_data()
num_factors = 50
model = CVAE(num_users=5551, num_items=16980, num_factors=num_factors, params=params, 
    input_dim=8000, dims=[200, 100], n_z=num_factors, activations=['sigmoid', 'sigmoid'], 
    loss_type='cross-entropy', lr=0.001, random_seed=0, print_step=10, verbose=False)
model.load_model(weight_path="model/pretrain")
model.run(data["train_users"], data["train_items"], data["test_users"], data["test_items"],
  data["content"], params)
model.save_model(weight_path="model/cvae", pmf_path="model/pmf")
