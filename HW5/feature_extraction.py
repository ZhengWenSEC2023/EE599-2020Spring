# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:01:05 2020

@author: Lenovo
"""

"""
This part is done in local computer
"""


import librosa
import os.path as osp
import os
import numpy as np
import h5py
from tensorflow.keras.layers import Input, Dense, GRU, BatchNormalization, Dropout
from tensorflow.keras import Model

eng_dir = 'D:\\EE599\\HW5\\train\\train_english'
chi_dir = 'D:\\EE599\\HW5\\train\\train_mandarin'
hin_dir = 'D:\\EE599\\HW5\\train\\train_hindi'

leng_seq = 2000
top_db = 35
val_ratio = 0.8

eng_list = os.listdir(eng_dir)
eng_respo = np.zeros((1, 64))
for each_name in eng_list:    
    y, sr = librosa.load(osp.join(eng_dir, each_name), sr=16000)
    intervals = librosa.effects.split(y, top_db=top_db)
    y_new = np.zeros((1))
    for interval in intervals:
        y_new = np.concatenate((y_new, y[interval[0]: interval[1]]))
    y = y_new[1:]
    mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
    mat = mat.T
    print(mat.shape)
    eng_respo = np.concatenate([eng_respo, mat], axis=0)
eng_respo = eng_respo[1:]
eng_respo = eng_respo[:5600000] # weight 4
# eng_respo = eng_respo[:7650000]
# eng_respo = eng_respo[:1450000] # balanced
# # eng_respo = eng_respo[:360000] # balanced_val
eng_label = 0 * np.ones((eng_respo.shape[0], 1))
eng_respo = np.reshape(eng_respo, (-1, leng_seq, 64))
eng_label = np.reshape(eng_label, (-1, leng_seq, 1))


hin_list = os.listdir(hin_dir)
hin_respo = np.zeros((1, 64))
for each_name in hin_list:    
    y, sr = librosa.load(osp.join(hin_dir, each_name), sr=16000)
    intervals = librosa.effects.split(y, top_db=top_db)
    y_new = np.zeros((1))
    for interval in intervals:
        y_new = np.concatenate((y_new, y[interval[0]: interval[1]]))
    y = y_new[1:]
    mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
    mat = mat.T
    print(mat.shape)
    hin_respo = np.concatenate([hin_respo, mat], axis=0)
hin_respo = hin_respo[1:]
hin_respo = hin_respo[:1400000] # weight 1
# # hin_respo = hin_respo[:2320000]
# hin_respo = hin_respo[:1450000]  # balanced
# # hin_respo = hin_respo[:360000]  # balanced_val
hin_label = 1 * np.ones((hin_respo.shape[0], 1))
hin_respo = np.reshape(hin_respo, (-1, leng_seq, 64))
hin_label = np.reshape(hin_label, (-1, leng_seq, 1))


chi_list = os.listdir(chi_dir)
chi_respo = np.zeros((1, 64))
for each_name in chi_list:    
    y, sr = librosa.load(osp.join(chi_dir, each_name), sr=16000)
    intervals = librosa.effects.split(y, top_db=top_db)
    y_new = np.zeros((1))
    for interval in intervals:
        y_new = np.concatenate((y_new, y[interval[0]: interval[1]]))
    y = y_new[1:]
    mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
    mat = mat.T
    print(mat.shape)
    chi_respo = np.concatenate([chi_respo, mat], axis=0)
chi_respo = chi_respo[1:]
chi_respo = chi_respo[:4200000] # weight 3
# chi_respo = chi_respo[:5340000]
# chi_respo = chi_respo[:1450000]  # balanced
# # chi_respo = chi_respo[:360000]  # balanced_val
chi_label = 2 * np.ones((chi_respo.shape[0], 1))
chi_respo = np.reshape(chi_respo, (-1, leng_seq, 64))
chi_label = np.reshape(chi_label, (-1, leng_seq, 1))

num_eng = int(5600000 / leng_seq * val_ratio)
num_hin = int(1400000 / leng_seq * val_ratio)
num_chi = int(4200000 / leng_seq * val_ratio)
train_data = np.concatenate((eng_respo[:num_eng], hin_respo[:num_hin], chi_respo[:num_chi]), axis=0)
train_label = np.concatenate((eng_label[:num_eng], hin_label[:num_hin], chi_label[:num_chi]), axis=0)
val_data = np.concatenate((eng_respo[num_eng:], hin_respo[num_hin:], chi_respo[num_chi:]), axis=0)
val_label = np.concatenate((eng_label[num_eng:], hin_label[num_hin:], chi_label[num_chi:]), axis=0)



# train_data = np.reshape(train_data, (-1, leng_seq, 64))
# train_label = np.reshape(train_label, (-1, leng_seq, 1))

idx = np.random.permutation(train_data.shape[0])
train_data = train_data[idx, :, :]
train_label = train_label[idx, :, :]

idx = np.random.permutation(val_data.shape[0])
val_data = val_data[idx, :, :]
val_label = val_label[idx, :, :]

f = h5py.File('E://train_set_weight.hdf5', 'w')
f['data'] = train_data
f['label'] = train_label
f.close()

f = h5py.File('E://val_set_weight.hdf5', 'w')
f['data'] = val_data
f['label'] = val_label
f.close()


# # # After shuffel
# # ######
# # train_data_shape = train_data.shape[1:]
# # train_in = Input(shape=train_data_shape)
# # training_in = Input(batch_shape=(None,train_seq_length,feature_dim)) this works too
# x = Dense(64, activation='relu')(train_in)
# x = GRU(256, return_sequences=True, stateful=False, )(x)
# x = BatchNormalization()(x)
# x = Dense(64, activation='relu')(x)
# x = Dropout(0.25)(x)
# train_out = Dense(3, activation='softmax')(x)

# training_model = Model(inputs=train_in, outputs=train_out)
# training_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
# training_model.summary()

# training_model.fit(train_data, train_label, batch_size=16, epochs=20)


# ##### define the streaming-infernece model
# streaming_in = Input(batch_shape=(1,None,feature_dim))  ## stateful ==> needs batch_shape specified
# foo = GRU(4, return_sequences=False, stateful=True )(streaming_in)
# streaming_pred = Dense(1)(foo)
# streaming_model = Model(inputs=streaming_in, outputs=streaming_pred)

# streaming_model.compile(loss='mean_squared_error', optimizer='adam')
# streaming_model.summary()

# ##### copy the weights from trained model to streaming-inference model
# training_model.save_weights('weights.hd5', overwrite=True)
# streaming_model.load_weights('weights.hd5')

