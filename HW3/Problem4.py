# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:47:36 2020

@author: Lenovo
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import regularizers

import h5py
import numpy as np
import matplotlib.pyplot as plt

## these could be read with an arg-parser

#### get the dataset
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# # train_images.shape is (60000, 28, 28)
# #test_images.shape (10000, 28, 28)
# num_pixels = 28 * 28 
# train_images = train_images.reshape( (60000, num_pixels) ).astype(np.float32) / 255.0
# test_images = test_images.reshape( (10000, num_pixels) ).astype(np.float32)  / 255.0

data = h5py.File('D:\\EE599\\HW2\\binary_random_sp2020.hdf5','r')

human = data['human'][:]
machine = data['machine'][:]

X = np.vstack( ( human, machine) )
y = np.append( 1 * np.ones(np.shape(human)[0]), 0 *  np.ones(np.shape(machine)[0]))[:, None]
train_set = np.concatenate((X, y), axis=1)
np.random.shuffle(train_set)
X = train_set[:, :-1]
y = train_set[:, -1]

reg_val = 1e-5
dropout_rate=  0.2
# this uses the Functional API for definning the model
nnet_inputs = Input(shape=(20,), name='data')
z = Dense(20, activation='relu', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val), name='hidden1')(nnet_inputs)
z = Dropout(dropout_rate)(z)
z = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val), name='output')(z)

our_first_model = Model(inputs=nnet_inputs, outputs=z)

#this will print a summary of the model to the screen
our_first_model.summary()

#this will produce a digram of the model -- requires pydot and graphviz installed
plot_model(our_first_model, to_file='our_first_model.png', show_shapes=True, show_layer_names=True)

our_first_model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])
results = our_first_model.fit(X, y, batch_size=16, epochs=40, validation_split=0.2)

loss = results.history['loss']
val_loss = results.history['val_loss']
acc = results.history['accuracy']
val_acc = results.history['val_accuracy']

epochs = np.arange(len(loss))

plt.figure()
plt.plot(epochs, loss, label='loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.xlabel('epochs')
plt.ylabel('Binary Cross Entropy Loss')
plt.title('Loss Curve')
plt.legend()

plt.figure()
plt.plot(epochs, acc, label='acc')
plt.plot(epochs, val_acc, label='val_acc')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

our_first_model.save('Problem4.hdf5')

train_para_output_W = our_first_model.get_layer('hidden1').get_weights()[0]
plt.figure()
plt.imshow(train_para_output_W, interpolation='nearest')
plt.show()

# using a .hdf5 or .h5 extension saves the model in format compatible with older keras

# train_para_hidden_W = our_first_model.get_layer('hidden').get_weights()[0]
# plt.figure()
# plt.hist(train_para_hidden_W)
# plt.xlabel('value of wieight')
# plt.ylabel('appearance time')
# plt.title('Weight Distribution of Hidden Layer')
# plt.show()

# train_para_output_W = our_first_model.get_layer('output').get_weights()[0]
# plt.figure()
# plt.hist(train_para_output_W)
# plt.xlabel('value of wieight')
# plt.ylabel('appearance time')
# plt.title('Weight Distribution of Output Layer')
# plt.show()