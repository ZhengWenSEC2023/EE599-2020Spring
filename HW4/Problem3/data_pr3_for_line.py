from torchvision import transforms
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os.path as osp
import json
from PIL import Image
import random

import tensorflow
from utils_pr3 import Config


class Combination:
    def __init__(self):
        random.seed(23)
        
        
    def combination(self, data_list, k):
        self.res = []
        if len(data_list) < 2:
            return self.res
        self.dfs(data_list, 0, k, [])
        return self.res
    
    
    def dfs(self, data_list, start, k, temp):
        if len(temp) == k:
            random.shuffle(temp)
            self.res.append(temp[:])  ## permutation invariant
            return
        
        for i in range(start, len(data_list)):
            if data_list[i] in temp:
                return
            temp.append(data_list[i])
            self.dfs(data_list, start + 1, k, temp)
            temp.pop()

    

class DataGenerator_test(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset, dataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.X, self.transform = dataset
        
        self.X1 = []
        self.X2 = []
        
        for each in self.X:
            self.X1.append(each[0])
            self.X2.append(each[1])
        
        self.image_dir = osp.join(Config['root_path'], 'images')
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
        X1, X2 = self.__data_generation(indexes)
        X1, X2 = np.stack(X1), np.stack(X2)
        return [np.moveaxis(X1, 1, 3), np.moveaxis(X2, 1, 3)]


    def __data_generation(self, indexes):
        X1 = []; X2 = []
        for idx in indexes:
            file_path1 = self.X1[idx]; file_path2 = self.X2[idx]; 
            X1.append(self.transform(Image.open(file_path1)))
            X2.append(self.transform(Image.open(file_path2)))
        return X1, X2


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X1))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class polyvore_dataset_test_line:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        self.meta_test_txt = open(osp.join(self.root_dir, Config['test_txt']))
        self.meta_test = open(osp.join(self.root_dir, Config['test_json']), 'r')
        self.meta_test = json.load(self.meta_test)
        
        meta_test_temp = {}
        for each in self.meta_test:
            meta_test_temp[each['set_id']] = each['items']
        self.meta_test = meta_test_temp.copy()
        
        for each_key in self.meta_test:
            each_key_temp = {}
            for each_dic in self.meta_test[each_key]:
                each_key_temp[each_dic['index']] = each_dic['item_id']
            self.meta_test[each_key] = each_key_temp
        
        
    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms


    def create_dataset_test_pairwise(self, line_test):
        line_test_data = line_test.split(' ')
        combs = Combination().combination(line_test_data, 2)
        pairs_test_X = []
        for comb in combs:
            set_id = [comb[0].split('_')[0], comb[1].split('_')[0]]
            index = [int(comb[0].split('_')[1]), int(comb[1].split('_')[1])]
            if set_id[0] in self.meta_test and set_id[1] in self.meta_test:
                pairs_test_X.append(
                        [self.image_dir + '/' + self.meta_test[set_id[0]][index[0]] + '.jpg', 
                         self.image_dir + '/' + self.meta_test[set_id[1]][index[1]] + '.jpg']
                        )

        return pairs_test_X

    
