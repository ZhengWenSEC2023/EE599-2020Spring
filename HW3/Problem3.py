# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:26:50 2020

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 22:13:54 2020

@author: Lenovo
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers

import numpy as np
import matplotlib.pyplot as plt

## these could be read with an arg-parser
reg_val = 0.0001
dropout_rate = 0.2

#### get the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# train_images.shape is (60000, 28, 28)
#test_images.shape (10000, 28, 28)
num_pixels = 28 * 28 
train_images = train_images.reshape( (60000, num_pixels) ).astype(np.float32) / 255.0
test_images = test_images.reshape( (10000, num_pixels) ).astype(np.float32)  / 255.0

# this uses the Functional API for definning the model
nnet_inputs = Input(shape=(num_pixels,), name='images')
z = Dense(48, activation='relu', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val), name='hidden')(nnet_inputs)
# z = Dropout(dropout_rate)(z)
z = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val), name='output')(z)

our_first_model = Model(inputs=nnet_inputs, outputs=z)

#this will print a summary of the model to the screen
our_first_model.summary()

#this will produce a digram of the model -- requires pydot and graphviz installed
plot_model(our_first_model, to_file='our_first_model.png', show_shapes=True, show_layer_names=True)

our_first_model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
results = our_first_model.fit(train_images,  train_labels, batch_size=32, epochs=40, validation_split=0.2)

# using a .hdf5 or .h5 extension saves the model in format compatible with older keras

train_para_hidden_W = our_first_model.get_layer('hidden').get_weights()[0]
plt.figure()
plt.hist(train_para_hidden_W)
plt.xlabel('value of wieight')
plt.ylabel('appearance time')
plt.title('Weight Distribution of Hidden Layer')
plt.show()

train_para_output_W = our_first_model.get_layer('output').get_weights()[0]
plt.figure()
plt.hist(train_para_output_W)
plt.xlabel('value of wieight')
plt.ylabel('appearance time')
plt.title('Weight Distribution of Output Layer')
plt.show()