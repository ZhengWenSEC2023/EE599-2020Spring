# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 19:34:39 2020

@author: Lenovo
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_div(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_div(x):
    return 1 - np.square(np.tanh(x))

def Softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)[:, None].T

def CrossEntropy(y, a):
    return -np.sum(y * np.log(a))

np.random.seed(23)

tndata = h5py.File('D:\\EE599\\HW3\\mnist_traindata.hdf5','r')
tnxdata = tndata['xdata'][:]
tnydata = tndata['ydata'][:]

train_set_num = 50000
data_set = np.concatenate((tnxdata, tnydata), axis=1)
np.random.shuffle(data_set)
train_set = data_set[:train_set_num]
train_data = train_set[:, :784].T
train_label = train_set[:, 784:].T
val_set = data_set[train_set_num:]
val_data = val_set[:, :784].T
val_label = val_set[:, 784:].T

# mini-batch
batch_size = 50
# learn rate init
eta = np.array([0.0001, 0.0005, 0.001])

# parameter init
# input layer
num_input = np.shape(train_data)[0]

# hidden: ReLU, tanh
num_layer_1 = 400
W1 = np.random.normal(0, 2 / num_input, (num_layer_1, num_input))
b1 = np.zeros((num_layer_1, 1))

num_layer_2 = 200
W2 = np.random.normal(0, 2 / num_layer_1, (num_layer_2, num_layer_1))
b2 = np.zeros((num_layer_2, 1))

num_output = 10
W3 = np.random.normal(0, 2 / num_layer_2, (num_output, num_layer_2))
b3 = np.zeros((num_output, 1))

epochs = 50
error_rate_train_epoch = []
error_rate_val_epoch = []

regular = 1e-5

act_func_hid = [ReLU, tanh]
act_func_div_hid = [ReLU_div, tanh_div]
for epoch in range(epochs):
    for iteration in range(train_set_num // batch_size):
        a0 = train_data[:, batch_size * iteration: batch_size * (iteration + 1)]
        a1 = act_func_hid[0](W1 @ a0 + b1)
        a2 = act_func_hid[0](W2 @ a1 + b2)
        a3 = Softmax(W3 @ a2 + b3)
        
        a2_div = act_func_div_hid[0](W2 @ a1 + b2)
        a1_div = act_func_div_hid[0](W1 @ a0 + b1)
        
        det3 = a3 - train_label[:, batch_size * iteration: batch_size * (iteration + 1)]
        det2 = a2_div * (W3.T @ det3)
        det1 = a1_div * (W2.T @ det2)
            
        W3 = W3 - eta[0] * (1 / batch_size) * det3 @ a2.T - 2 * regular * W3
        W2 = W2 - eta[0] * (1 / batch_size) * det2 @ a1.T - 2 * regular * W2
        W1 = W1 - eta[0] * (1 / batch_size) * det1 @ a0.T - 2 * regular * W1
        b3 = b3 - eta[0] * (1 / batch_size) * np.sum(det3, axis=1)[:, None]
        b2 = b2 - eta[0] * (1 / batch_size) * np.sum(det2, axis=1)[:, None]
        b1 = b1 - eta[0] * (1 / batch_size) * np.sum(det1, axis=1)[:, None]
                
    a0_train = train_data[:]
    a1_train = act_func_hid[0](W1 @ a0_train + b1)
    a2_train = act_func_hid[0](W2 @ a1_train + b2)
    a3_train = Softmax(W3 @ a2_train + b3)
    train_pred = (a3_train == np.max(a3_train, axis=0)[None, :]).astype(float)
    
    error_train = 0;
    for j in range(train_set_num):
        if (train_pred[:, j] != train_label[:, j]).any():
            error_train += 1
    error_rate_train = error_train / train_set_num
    error_rate_train_epoch.append(error_rate_train)
    
    a0_val = val_data[:]
    a1_val = act_func_hid[0](W1 @ a0_val + b1)
    a2_val = act_func_hid[0](W2 @ a1_val + b2)
    a3_val = Softmax(W3 @ a2_val + b3)
    val_pred = (a3_val == np.max(a3_val, axis=0)[None, :]).astype(float)
    
    error_val = 0;
    for j in range(60000 - train_set_num):
        if (val_pred[:, j] != val_label[:, j]).any():
            error_val += 1
    error_rate_val = error_val / (60000 - train_set_num)
    error_rate_val_epoch.append(error_rate_val)
    
    if epoch == 15:
        eta /= 2
    if epoch == 30:
        eta /= 2
        
    print('Epoch', epoch, 'train error', error_rate_train, 'val error', error_rate_val)
        
tsdata = h5py.File('D:\\EE599\\HW3\\mnist_testdata.hdf5','r')
tsxdata = tsdata['xdata'][:].T
tsydata = tsdata['ydata'][:].T

a0_test = tsxdata[:]
a1_test = act_func_hid[0](W1 @ a0_test + b1)
a2_test = act_func_hid[0](W2 @ a1_test + b2)
a3_test = Softmax(W3 @ a2_test + b3)
test_pred = (a3_test == np.max(a3_test, axis=0)[None, :]).astype(float)
test_label = tsydata[:]
error_test = 0
for j in range(np.shape(tsxdata)[1]):
    if (test_pred[:, j] != test_label[:, j]).any():
        error_test += 1
error_rate_test = error_test / np.shape(tsxdata)[1]

accuracy_train_epoch = 1 - np.array(error_rate_train_epoch)
accuracy_val_epoch = 1 - np.array(error_rate_val_epoch)
accuracy_test = 1 - error_rate_test


plt.figure()
plt.plot(accuracy_train_epoch, label='train')
plt.plot(accuracy_val_epoch, label='val')
plt.plot(15, accuracy_train_epoch[15], 'bo')
plt.plot(30, accuracy_train_epoch[30], 'bo')
plt.plot(15, accuracy_val_epoch[15], 'ro')
plt.plot(30, accuracy_val_epoch[30], 'ro')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('batch size = 50, act = ReLU, eta = 0.0001')
plt.legend()


## 
tndata = h5py.File('D:\\EE599\\HW3\\mnist_traindata.hdf5','r')
tnxdata = tndata['xdata'][:]
tnydata = tndata['ydata'][:]

data_set = np.concatenate((tnxdata, tnydata), axis=1)
np.random.shuffle(data_set)
train_set = data_set[:]
train_data = train_set[:, :784].T
train_label = train_set[:, 784:].T
# mini-batch
batch_size = 50
# learn rate init
eta = np.array([0.0001, 0.0005, 0.001])
# parameter init
# input layer
num_input = np.shape(train_data)[0]

# hidden: ReLU, tanh
num_layer_1 = 400
W1 = np.random.normal(0, 2 / num_input, (num_layer_1, num_input))
b1 = np.zeros((num_layer_1, 1))

num_layer_2 = 200
W2 = np.random.normal(0, 2 / num_layer_1, (num_layer_2, num_layer_1))
b2 = np.zeros((num_layer_2, 1))

num_output = 10
W3 = np.random.normal(0, 2 / num_layer_2, (num_output, num_layer_2))
b3 = np.zeros((num_output, 1))

epochs = 50
error_rate_train_epoch = []
error_rate_val_epoch = []

regular = 1e-5
train_set_num = 60000
act_func_hid = [ReLU, tanh]
act_func_div_hid = [ReLU_div, tanh_div]
for epoch in range(epochs):
    for iteration in range(train_set_num // batch_size):
        a0 = train_data[:, batch_size * iteration: batch_size * (iteration + 1)]
        a1 = act_func_hid[0](W1 @ a0 + b1)
        a2 = act_func_hid[0](W2 @ a1 + b2)
        a3 = Softmax(W3 @ a2 + b3)
        
        a2_div = act_func_div_hid[0](W2 @ a1 + b2)
        a1_div = act_func_div_hid[0](W1 @ a0 + b1)
        
        det3 = a3 - train_label[:, batch_size * iteration: batch_size * (iteration + 1)]
        det2 = a2_div * (W3.T @ det3)
        det1 = a1_div * (W2.T @ det2)
            
        W3 = W3 - eta[2] * (1 / batch_size) * det3 @ a2.T - 2 * regular * W3
        W2 = W2 - eta[2] * (1 / batch_size) * det2 @ a1.T - 2 * regular * W2
        W1 = W1 - eta[2] * (1 / batch_size) * det1 @ a0.T - 2 * regular * W1
        b3 = b3 - eta[2] * (1 / batch_size) * np.sum(det3, axis=1)[:, None]
        b2 = b2 - eta[2] * (1 / batch_size) * np.sum(det2, axis=1)[:, None]
        b1 = b1 - eta[2] * (1 / batch_size) * np.sum(det1, axis=1)[:, None]
                
    a0_train = train_data[:]
    a1_train = act_func_hid[0](W1 @ a0_train + b1)
    a2_train = act_func_hid[0](W2 @ a1_train + b2)
    a3_train = Softmax(W3 @ a2_train + b3)
    train_pred = (a3_train == np.max(a3_train, axis=0)[None, :]).astype(float)
    
    error_train = 0;
    for j in range(train_set_num):
        if (train_pred[:, j] != train_label[:, j]).any():
            error_train += 1
    error_rate_train = error_train / train_set_num
    error_rate_train_epoch.append(error_rate_train)
    if epoch == 15:
        eta /= 2
    if epoch == 30:
        eta /= 2
        
    print('Epoch', epoch, 'train error', error_rate_train)

tsdata = h5py.File('D:\\EE599\\HW3\\mnist_testdata.hdf5','r')
tsxdata = tsdata['xdata'][:].T
tsydata = tsdata['ydata'][:].T

a0_test = tsxdata[:]
a1_test = act_func_hid[0](W1 @ a0_test + b1)
a2_test = act_func_hid[0](W2 @ a1_test + b2)
a3_test = Softmax(W3 @ a2_test + b3)
test_pred = (a3_test == np.max(a3_test, axis=0)[None, :]).astype(float)
test_label = tsydata[:]
error_test = 0
for j in range(np.shape(tsxdata)[1]):
    if (test_pred[:, j] != test_label[:, j]).any():
        error_test += 1
error_rate_test = error_test / np.shape(tsxdata)[1]
