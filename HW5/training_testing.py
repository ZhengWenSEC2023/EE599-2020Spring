# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 23:43:38 2020

@author: Lenovo
"""

""" 
This part is done in Colab
"""

import numpy as np
import h5py
from tensorflow.keras.layers import Input, Dense, GRU, BatchNormalization, Dropout
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import librosa


def create_weights_matrix(labels): 
    weights_mapping = {0:0.75, 1:3, 2:1}
    weights_matrix = np.zeros(labels.shape[0:2])
    for i,sample in enumerate(labels):
        for j,elem in enumerate(sample):
            weights_matrix[i,j] = weights_mapping[elem[0]]
    return weights_matrix

# load dataset, generate weight, define training model
!gdown https://drive.google.com/uc?id=1e0PEPnA7N6kIGC0mHv5QQOgB_7OuFGF7 # train
!gdown https://drive.google.com/uc?id=1G2z2c4eeW0IB-rxiLfkuTyfpmkfCq9kL # test

f = h5py.File('/content/train_set_weight.hdf5', 'r')
train_data = f['data'].value
train_label = f['label'].value
f.close()

f = h5py.File('/content/val_set_weight.hdf5', 'r')
val_data = f['data'].value
val_label = f['label'].value
f.close()

weight_matrix = create_weights_matrix(train_label)

train_data_shape = train_data.shape[1:]
train_in = Input(shape=train_data_shape)
x = Dense(128, activation='relu')(train_in)
x = GRU(256, 
        return_sequences=True, 
        stateful=False, 
        kernel_regularizer=regularizers.l2(0.001),
        recurrent_regularizer=regularizers.l2(0.001),
        dropout=0.4)(x)
x = BatchNormalization()(x)
x = GRU(256, 
        return_sequences=True, 
        stateful=False, 
        kernel_regularizer=regularizers.l2(0.001),
        recurrent_regularizer=regularizers.l2(0.001),
        dropout=0.4)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
train_out = Dense(3, activation='softmax')(x)
training_model = Model(inputs=train_in, outputs=train_out)
training_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'], sample_weight_mode = 'temporal')
training_model.summary()
plot_model(streaming_model, to_file='streaming_model.png')
# training
history = training_model.fit(train_data, 
                             train_label, 
                             batch_size=16, 
                             epochs=30, 
                             validation_data=(val_data, val_label), 
                             sample_weight=weight_matrix                             
                             )

training_model.save_weights('weights_0_30_silence_balance.hd5', overwrite=True)

# define streaming model
streaming_in_shape = train_data.shape[2:]
streaming_in = Input(batch_shape=(1,None,streaming_in_shape[0]))  ## stateful ==> needs batch_shape specified
x = Dense(128, activation='relu')(streaming_in)
x = GRU(256, 
        return_sequences=True, 
        stateful=True, 
        kernel_regularizer=regularizers.l2(0.001),
        recurrent_regularizer=regularizers.l2(0.001),
        dropout=0.4)(x)
x = BatchNormalization()(x)
x = GRU(256, 
        return_sequences=False, 
        stateful=True, 
        kernel_regularizer=regularizers.l2(0.001),
        recurrent_regularizer=regularizers.l2(0.001),
        dropout=0.4)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
streaming_out = Dense(3, activation='softmax')(x)
streaming_model = Model(inputs=streaming_in, outputs=streaming_out)
streaming_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
streaming_model.summary()
streaming_model.load_weights('weights_0_30_silence_balance.hd5')

# save training history
f = h5py.File('train_his_0_30.h5py', 'w')
f['acc'] = history.history['accuracy']
f['val_acc'] = history.history['val_accuracy']
f['loss'] = history.history['loss']
f['val_loss'] = history.history['val_loss']
f.close()

# load data and plot
f = h5py.File('train_his_0_30.h5py', 'r')
plt.figure()
plt.plot(f['acc'], label='train')
plt.plot(f['val_acc'], label='val')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(f['loss'])
plt.plot(f['val_loss'])
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Multiclass Cross Entropy')

# testing
y, sr = librosa.load('/content/english_0117.wav', sr=16000)
mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
mat = mat.T
print(mat.shape)
in_seq = mat.reshape( (1, -1, streaming_in_shape[0]) )
in_seq = in_seq[:, :2000, :64]
streaming_model.reset_states()
pred = np.zeros((1, 3))
for n in range(in_seq.shape[1]):
    in_feature_vector = in_seq[0][n].reshape(1,-1, in_seq.shape[2])
    single_pred = streaming_model.predict(in_feature_vector)
    pred = np.concatenate([pred, np.squeeze(single_pred)[None, :]])
    if n % 1000 == 0:
      print(n)
plt.plot(pred[:, 0], label='Eng')
plt.plot(pred[:, 1], label='Hin')
plt.plot(pred[:, 2], label='Chi')
plt.title('english_0117.wav')
plt.ylabel('Probability')
plt.xlabel('Slice')
plt.legend()

y, sr = librosa.load('/content/hindi_0033.wav', sr=16000)
mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
mat = mat.T
print(mat.shape)
in_seq = mat.reshape( (1, -1, streaming_in_shape[0]) )
in_seq = in_seq[:, :2000, :64]
streaming_model.reset_states()
pred = np.zeros((1, 3))
for n in range(in_seq.shape[1]):
    in_feature_vector = in_seq[0][n].reshape(1,-1, in_seq.shape[2])
    single_pred = streaming_model.predict(in_feature_vector)
    pred = np.concatenate([pred, np.squeeze(single_pred)[None, :]])
    if n % 1000 == 0:
      print(n)
plt.plot(pred[:, 0], label='Eng')
plt.plot(pred[:, 1], label='Hin')
plt.plot(pred[:, 2], label='Chi')
plt.title('hindi_0033.wav')
plt.ylabel('Probability')
plt.xlabel('Slice')
plt.legend()

y, sr = librosa.load('/content/mandarin_0085.wav', sr=16000)
mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
mat = mat.T
print(mat.shape)
in_seq = mat.reshape( (1, -1, streaming_in_shape[0]) )
in_seq = in_seq[:, :2000, :64]
streaming_model.reset_states()
pred = np.zeros((1, 3))
for n in range(in_seq.shape[1]):
    in_feature_vector = in_seq[0][n].reshape(1,-1, in_seq.shape[2])
    single_pred = streaming_model.predict(in_feature_vector)
    pred = np.concatenate([pred, np.squeeze(single_pred)[None, :]])
    if n % 1000 == 0:
      print(n)
plt.plot(pred[:, 0], label='Eng')
plt.plot(pred[:, 1], label='Hin')
plt.plot(pred[:, 2], label='Chi')
plt.title('mandarin_0085.wav')
plt.ylabel('Probability')
plt.xlabel('Slice')
plt.legend()