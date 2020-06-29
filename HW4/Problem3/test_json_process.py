# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:13:47 2020

@author: Lenovo
"""

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, add, BatchNormalization, concatenate
from tensorflow.keras.layers import Conv2D, Input, Dropout, MaxPooling2D
from tensorflow.keras.models import Model
from data_pr3_for_line import polyvore_dataset_test_line, DataGenerator_test
from utils_pr3 import Config
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import h5py
from tensorflow.keras.utils import plot_model
import os.path as osp
import numpy as np


if __name__=='__main__':
    
    root_dir = Config['root_path']
    meta_test_txt = open(osp.join(root_dir, Config['test_txt_group']))
    
    model_path = 'pr3Mod.hdf5'
    
    dataset = polyvore_dataset_test_line()
    transforms = dataset.get_data_transforms()
    
    params = {'batch_size': Config['batch_test'],
              'shuffle': True}
    
    model = tf.keras.models.load_model(model_path)
    
    f = open('pred_res_Extra.txt', 'w')
    
    line_test = ' '.join(meta_test_txt.readline().split())
    
    i = 1
    while line_test:
        X_test_pairwise = dataset.create_dataset_test_pairwise(line_test)
        test_set_pairwise = (X_test_pairwise, transforms['test'])
        dataset_size = {'test': len(X_test_pairwise)}
        test_generator = DataGenerator_test(test_set_pairwise, dataset_size, params)
        predictions = model.predict_generator(generator=test_generator)

        final_prediction = np.argmax(np.average(predictions, axis=0))
        
        f.write(str(final_prediction) + ' ' + line_test)
        f.write('\n')
        
        print(i)
        
        line_test = ' '.join(meta_test_txt.readline().split())
        
        i += 1
        
    
    f.close()
    
