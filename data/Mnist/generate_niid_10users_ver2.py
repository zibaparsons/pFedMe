from sklearn.datasets import fetch_mldata
from scipy.io import loadmat
from tqdm import trange
import numpy as np
import random
import json
import os

random.seed(1)
np.random.seed(1)
files = os.listdir(os.curdir)
print(files)
NUM_USERS = 10
NUM_LABELS = 2
# Setup directory for train/test data
train_path = './data/train/mnist_train.json'
test_path = './data/test/mnist_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get MNIST data, normalize, and divide by level
#mnist = fetch_mldata('MNIST original', data_home='./data')
#mnist = loadmat('mnist-original.mat')
mnist = fetch_mldata('MNIST original', data_home='./data')
mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)

mnist_data_train = mnist.data[0:60000].copy()
mnist_target_train  = mnist.target[0:60000].copy()

mnist_data_test  = mnist.data[60000:].copy()
mnist_target_test = mnist.target[60000:].copy()

mnist_data = []
for i in trange(10):
    idx = mnist_target_train==i
    mnist_data.append(mnist_data_train[idx])

print("\nNumb samples of each label:\n", [len(v) for v in mnist_data])
users_lables = []

print("idx",idx)
# devide for label for each users:
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        l = (user + j) % 10
        users_lables.append(l)
unique, counts = np.unique(users_lables, return_counts=True)
print("--------------")
print(unique, counts)

def ram_dom_gen(total, size):
    print(total)
    nums = []
    temp = []
    for i in range(size - 1):
        val = np.random.randint(total//(size + 1), total//2)
        temp.append(val)
        total -= val
    temp.append(total)
    print(temp)
    return temp
number_sampe = []
for total_value, count in zip(mnist_data, counts):
    temp = ram_dom_gen(len(total_value), count)
    number_sampe.append(temp)
print("--------------")
print(number_sampe)

i = 0
number_samples = []
for i in range(2):
    for sample in number_sampe:
        print(sample)
        number_samples.append(sample[i])

print(number_samples)

###### CREATE USER DATA SPLIT #######
# Assign 100 samples to each user
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
count = 0
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        l = (user + j) % 10
        print("value of L",l)
        num_samples =  number_samples[count] # num sample
        count = count + 1
        if idx[l] + num_samples < len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:num_samples].tolist()
            y[user] += (l*np.ones(num_samples)).tolist()
            idx[l] += num_samples
            print("check len os user:", user, j,"len data", len(X[user]), num_samples)

print("IDX2:", idx) # counting samples for each labels

# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

# Setup 5 users
# for i in trange(5, ncols=120):
test_size = ram_dom_gen(10000, NUM_USERS)
index_data = 0
combined_test = list(zip(mnist_data_test.tolist(),mnist_target_test.tolist()))
random.shuffle(combined_test)
X_test = []
y_test = []
X_test[:], y_test[:] = zip(*combined_test)
for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)
    
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)

    num_samples = len(X[i])
    train_len = int(num_samples)
    
    
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)

    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X_test[index_data: index_data + test_size[i]], 'y': y_test[index_data: index_data+ test_size[i]]}
    test_data['num_samples'].append(test_size[i])
    index_data =  index_data + test_size[i]

print("index_data", index_data)
#test_size = ram_dom_gen(10000, NUM_USERS)

print("Num_samples:", train_data['num_samples'])
print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")