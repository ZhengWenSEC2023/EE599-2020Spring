import numpy as np
import os
import os.path as osp
import argparse

Config ={}
# you should replace it with your own root_path
# Config['root_path'] = 'D:\EE599\HW4\polyvore_outfits'
Config['root_path'] = '/content/polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = ''
Config['meta_train_txt'] = 'compatibility_train.txt'
Config['meta_val_txt'] = 'compatibility_valid.txt'


Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 10


Config['batch_size'] = 128

Config['learning_rate'] = 0.001
Config['num_workers'] = 1

