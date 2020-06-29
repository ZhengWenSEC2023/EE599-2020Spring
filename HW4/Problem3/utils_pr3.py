import numpy as np
import os
import os.path as osp
import argparse

Config ={}
# you should replace it with your own root_path
Config['root_path'] = '/content/polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'

Config['train_json'] = 'train.json'
Config['val_json'] = 'valid.json'


Config['checkpoint_path'] = ''
Config['train_txt'] = 'compatibility_train.txt'
Config['val_txt'] = 'compatibility_valid.txt'
Config['test_txt'] = 'test_pairwise_compat_hw.txt'

Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 20


Config['batch_size'] = 1

Config['learning_rate'] = 0.002
Config['num_workers'] = 1

Config['num_train'] = 180000
Config['num_val'] = 20000

Config['test_txt_group'] = 'compatibility_test_hw.txt'
Config['test_json'] = 'test.json'
Config['batch_test'] = 1