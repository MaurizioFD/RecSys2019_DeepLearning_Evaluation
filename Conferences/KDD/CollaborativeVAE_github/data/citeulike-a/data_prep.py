import numpy as np
pairs_train = []
pairs_test = []
num_train_per_user = 10
user_id = 0
np.random.seed(123)
for line in open("users.dat"):
	arr = line.strip().split()
	arr = np.asarray([int(x) for x in arr[1:]])
	n = len(arr)
	idx = np.random.permutation(n)
	# assert(n > num_train_per_user)
	for i in range(min(num_train_per_user, n)):
		pairs_train.append((user_id, arr[idx[i]]))
	if n > num_train_per_user:
		for i in range(num_train_per_user, n):
			pairs_test.append((user_id, arr[idx[i]]))
	user_id += 1
num_users = user_id

# pairs_train and  pairs_test are list of tuples (user_id, item)
pairs_train = np.asarray(pairs_train)
pairs_test = np.asarray(pairs_test)
print(pairs_train.dtype)
print(pairs_test.dtype)
num_items = np.maximum(np.max(pairs_train[:, 1]), np.max(pairs_test[:, 1]))+1
print("num_users=%d, num_items=%d" % (num_users, num_items))

with open("cf-train-"+str(num_train_per_user)+"-users.dat", "w") as fid:
	for user_id in range(num_users):
		this_user_items = pairs_train[pairs_train[:, 0]==user_id, 1]
		items_str = " ".join(str(x) for x in this_user_items)
		fid.write("%d %s\n" % (len(this_user_items), items_str))

with open("cf-train-"+str(num_train_per_user)+"-items.dat", "w") as fid:
	for item_id in range(num_items):
		this_item_users = pairs_train[pairs_train[:, 1]==item_id, 0]
		users_str = " ".join(str(x) for x in this_item_users)
		fid.write("%d %s\n" % (len(this_item_users), users_str))

with open("cf-test-"+str(num_train_per_user)+"-users.dat", "w") as fid:
	for user_id in range(num_users):
		this_user_items = pairs_test[pairs_test[:, 0]==user_id, 1]
		items_str = " ".join(str(x) for x in this_user_items)
		fid.write("%d %s\n" % (len(this_user_items), items_str))

with open("cf-test-"+str(num_train_per_user)+"-items.dat", "w") as fid:
	for item_id in range(num_items):
		this_item_users = pairs_test[pairs_test[:, 1]==item_id, 0]
		users_str = " ".join(str(x) for x in this_item_users)
		fid.write("%d %s\n" % (len(this_item_users), users_str))
