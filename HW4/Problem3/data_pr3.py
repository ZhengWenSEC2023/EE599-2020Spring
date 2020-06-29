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


class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()



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



    def create_dataset_train(self):
        # map id to category
        meta_file = open(osp.join(self.root_dir, Config['meta_file']), 'r')
        meta_json = json.load(meta_file)
        meta_train_txt = open(osp.join(self.root_dir, Config['train_txt']))
        meta_val_txt = open(osp.join(self.root_dir, Config['val_txt']))
        
        # train

        meta_train = open(osp.join(self.root_dir, Config['train_json']), 'r')
        meta_train = json.load(meta_train)
        meta_train_temp = {}
        for each in meta_train:
            meta_train_temp[each['set_id']] = each['items']
        meta_train = meta_train_temp.copy()
        
        for each_key in meta_train:
            each_key_temp = {}
            for each_dic in meta_train[each_key]:
                each_key_temp[each_dic['index']] = each_dic['item_id']
            meta_train[each_key] = each_key_temp
            
        pairs_train_X = []
        pairs_train_y = []
        line_train = ' '.join(meta_train_txt.readline().split())
        while line_train:
            line_train_set = line_train.split(' ')
            line_train_label = line_train_set[0]
            line_train_data = line_train_set[1: ]
            combs = Combination().combination(line_train_data, 2)
            for comb in combs:
                set_id = [comb[0].split('_')[0], comb[1].split('_')[0]]
                index = [int(comb[0].split('_')[1]), int(comb[1].split('_')[1])]
                if set_id[0] in meta_train and set_id[1] in meta_train:
                    pairs_train_X.append(
                            [self.image_dir + '/' + meta_train[set_id[0]][index[0]] + '.jpg', 
                             self.image_dir + '/' + meta_train[set_id[1]][index[1]] + '.jpg']
                            )
                    pairs_train_y.append(int(line_train_label))
                        
            line_train = ' '.join(meta_train_txt.readline().split())

        
        #  val 
        
        meta_val = open(osp.join(self.root_dir, Config['val_json']), 'r')
        meta_val = json.load(meta_val)
        meta_val_temp = {}
        for each in meta_val:
            meta_val_temp[each['set_id']] = each['items']
        meta_val = meta_val_temp.copy()
        
        for each_key in meta_val:
            each_key_temp = {}
            for each_dic in meta_val[each_key]:
                each_key_temp[each_dic['index']] = each_dic['item_id']
            meta_val[each_key] = each_key_temp
            
        pairs_val_X = []
        pairs_val_y = []
        line_val = ' '.join(meta_val_txt.readline().split())
        while line_val:
            line_val_set = line_val.split(' ')
            line_val_label = line_val_set[0]
            line_val_data = line_val_set[1: ]
            combs = Combination().combination(line_val_data, 2)
            for comb in combs:
                set_id = [comb[0].split('_')[0], comb[1].split('_')[0]]
                index = [int(comb[0].split('_')[1]), int(comb[1].split('_')[1])]
                if set_id[0] in meta_val and set_id[1] in meta_val:
                    pairs_val_X.append(
                        [self.image_dir + '/' + meta_val[set_id[0]][index[0]] + '.jpg', 
                         self.image_dir + '/' + meta_val[set_id[1]][index[1]] + '.jpg']
                        )
                    pairs_val_y.append(int(line_val_label))
                            
            line_val = ' '.join(meta_val_txt.readline().split())



        # split dataset
        X_train, X_val = pairs_train_X, pairs_val_X
        y_train, y_val = pairs_train_y, pairs_val_y
        return X_train, X_val, y_train, y_val


    def create_dataset_test(self):
        # map id to category
        meta_file = open(osp.join(self.root_dir, Config['meta_file']), 'r')
        meta_json = json.load(meta_file)
        meta_test_txt = open(osp.join(self.root_dir, Config['test_txt']))
        
        # test
            
        pairs_test_X = []
        line_test = ' '.join(meta_test_txt.readline().split())
        while line_test:
            line_test_data = line_test.split(' ')
            pairs_test_X.append(
                    [self.image_dir + '/' + line_test_data[0] + '.jpg', 
                     self.image_dir + '/' + line_test_data[1] + '.jpg']
                    )                
            line_test = ' '.join(meta_test_txt.readline().split())
            
        X_test = pairs_test_X
        return X_test


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, dataset, dataset_size, params):
        self.batch_size = params['batch_size']
        self.shuffle = params['shuffle']
        self.X, self.y, self.transform = dataset
        
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
        X1, X2, y = self.__data_generation(indexes)
        X1, X2, y = np.stack(X1), np.stack(X2), np.stack(y)
        return [np.moveaxis(X1, 1, 3), np.moveaxis(X2, 1, 3)], tensorflow.keras.utils.to_categorical(y, num_classes=2)


    def __data_generation(self, indexes):
        X1 = []; X2 = []; y = []
        for idx in indexes:
            file_path1 = self.X1[idx]; file_path2 = self.X2[idx]; 
            X1.append(self.transform(Image.open(file_path1)))
            X2.append(self.transform(Image.open(file_path2)))
            y.append(self.y[idx])
        return X1, X2, y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



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






