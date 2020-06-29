# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:24:50 2020

@author: Lenovo
"""
from tensorflow.keras.models import Model
from data_pr3 import polyvore_dataset, DataGenerator, DataGenerator_test
from utils_pr3 import Config
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import h5py
from tensorflow.keras.utils import plot_model
import numpy as np


dataset = polyvore_dataset()
transforms = dataset.get_data_transforms()
X_test = dataset.create_dataset_test()
test_set = (X_test, transforms['test'])
dataset_size = {'test': len(X_test)}
params = {'batch_size': Config['batch_size'], 'shuffle': True}
test_generator = DataGenerator_test(test_set, dataset_size, params)

model_path = 'pr3Mod.hdf5'
model = tf.keras.models.load_model(model_path)
predictions = model.predict(test_generator, verbose=1)

score = np.around(predictions[:, 1], decimals=2)

txt = open('test_pairwise_compat_hw.txt', 'r')

f = open('pair compatibility.txt', 'w')

line = ' '.join(txt.readline().split())
i = 0
while line:
    f.write(str(score[i]) + ' ' + line)
    f.write('\n')
    line = ' '.join(txt.readline().split())
    i += 1
    
f.close()
txt.close()

        