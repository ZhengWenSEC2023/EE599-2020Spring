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
reg_val = 0
dropout_rate = 0

#### get the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# train_images.shape is (60000, 28, 28)
#test_images.shape (10000, 28, 28)
num_pixels = 28 * 28 
train_images = train_images.reshape( (60000, num_pixels) ).astype(np.float32) / 255.0
test_images = test_images.reshape( (10000, num_pixels) ).astype(np.float32)  / 255.0

# this uses the Functional API for definning the model
nnet_inputs = Input(shape=(num_pixels,), name='images')
z = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val), name='hidden')(nnet_inputs)
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
our_first_model.save('fmnist_trained.hdf5')

# # plot our learning curves
# #results.history is a dictionary
# loss = results.history['loss']
# val_loss = results.history['val_loss']
# acc = results.history['accuracy']
# val_acc = results.history['val_accuracy']

# epochs = np.arange(len(loss))

# plt.figure()
# plt.plot(epochs, loss, label='loss')
# plt.plot(epochs, val_loss, label='val_loss')
# plt.xlabel('epochs')
# plt.ylabel('Multiclass Cross Entropy Loss')
# plt.title(f'Loss with Regularizer: {reg_val : 3.2g}; Dropout: {dropout_rate : 3.2g} ')
# plt.legend()
# plt.savefig(f'learning_loss_R_{reg_val}_D_{dropout_rate}.png', dpi=256)

# plt.figure()
# plt.plot(epochs, acc, label='acc')
# plt.plot(epochs, val_acc, label='val_acc')
# plt.xlabel('epochs')
# plt.ylabel('Accuracy')
# plt.title(f'Accuracy with Regularizer: {reg_val : 3.2g}; Dropout: {dropout_rate : 3.2g} ')
# plt.legend()
# plt.savefig(f'learning_acc_R_{reg_val}_D_{dropout_rate}.png', dpi=256)

# # read back out model, just to illustrate
# model_copy = load_model('fmnist_trained.hdf5')

# # perform inference on a single image:
# prediction = model_copy.predict(test_images.reshape( (10000, num_pixels) ) )
# predicted_label = np.argmax(prediction, axis=1)
# num_classes = 10
# class_decision = np.argmax(prediction.T, axis=0)

# # for m in range(num_classes):
# # 	if m == class_decision:
# # 		print(f'class{m}:\tclass soft-decisions:{prediction[m]}\t(hard decision)')
# # 	else:
# # 		print(f'class{m}:\tclass soft-decisions:{prediction[m]}')

# # test_loss, test_acc = model_copy.evaluate(test_images,  test_labels, verbose=2)
# # print(f'Test Loss: {test_loss : 3.2f}')
# # print(f'Test Accuracy: {100 * test_acc : 3.2f}%')

# confusion_matrix = tf.math.confusion_matrix( test_labels, class_decision )._numpy()
# plt.imshow(confusion_matrix, interpolation='nearest')
# plt.xlabel('Predicted Label')
# plt.ylabel('Test Label')

