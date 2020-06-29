import os.path as osp
import os
import numpy as np
import json


root_path = 'D:\\EE599\\HW4\\polyvore_outfits'
submission_dir_pr2_my = 'D:\\EE599\\HW4\\Submission\\Problem2\\mynet'
submission_dir_pr2_mob = 'D:\\EE599\\HW4\\Submission\\Problem2\\Mobnet'
meta_file = 'polyvore_item_metadata.json'
test_json = 'test.json'
txt_pr2_name = 'category.txt'

## category
pr2_test_my = open(osp.join(submission_dir_pr2_my, txt_pr2_name), 'r')
pr2_test_mob = open(osp.join(submission_dir_pr2_mob, txt_pr2_name), 'r')
meta_file = open(osp.join(root_path, meta_file), 'r')
meta_json = json.load(meta_file)
id_to_category = {}
for k, v in meta_json.items():
    id_to_category[k] = v['category_id']

## my
line = ' '.join(pr2_test_my.readline().split())
total_pr2 = 0
correct_my = 0
while line:
    image_id = line.split(' ')[0]
    image_cate_id = line.split(' ')[1]
    if meta_json[image_id]['category_id'] == image_cate_id:
        correct_my += 1
    total_pr2 += 1
    line = ' '.join(pr2_test_my.readline().split())
    
line = ' '.join(pr2_test_my.readline().split())

## mob
line = ' '.join(pr2_test_mob.readline().split())
total_pr2 = 0
correct_mob = 0
unique = set()
while line:
    image_id = line.split(' ')[0]
    image_cate_id = line.split(' ')[1]
    unique.add(meta_json[image_id]['category_id'])
    if meta_json[image_id]['category_id'] == image_cate_id:
        correct_mob += 1
    total_pr2 += 1
    line = ' '.join(pr2_test_mob.readline().split())
    
## compatibility 
submission_dir_pr3 = 'D:\\EE599\\HW4\\Submission\\Problem3'
score_res = 'pair_compatibility.txt'
set_res = 'outfit_compatibility.txt'

pr3_test = open(osp.join(submission_dir_pr3, score_res), 'r')
meta_file = open(osp.join(root_path, test_json), 'r')
meta_test = json.load(meta_file)
meta_test_temp = {}
for each in meta_test:
    meta_test_temp[each['set_id']] = each['items']
meta_test = meta_test_temp.copy()

meta_test_temp = {}
for each_set in meta_test:
    for each_item in meta_test[each_set]:
        meta_test_temp[each_item['item_id']] = each_set
meta_test = meta_test_temp.copy()


line = ' '.join(pr3_test.readline().split())
total_pr3 = 0
correct_pr3 = 0
while line:
    score = line.split(' ')[0]
    image_id_1 = line.split(' ')[1]
    image_id_2 = line.split(' ')[2]
    if meta_test[image_id_1] == meta_test[image_id_2] and float(score) > 0.5:
        correct_pr3 += 1
    if meta_test[image_id_1] != meta_test[image_id_2] and float(score) <= 0.5:
        correct_pr3 += 1
    total_pr3 += 1
    line = ' '.join(pr3_test.readline().split())
    
## compatibility extra
pr3_test_ex = open(osp.join(submission_dir_pr3, set_res), 'r')
line = ' '.join(pr3_test_ex.readline().split())
total_pr3_ex = 0
correct_pr3_ex = 0
while line:
    label = line.split(' ')[0]
    image_set = line.split(' ')[1:]
    unique = set()
    for each in image_set:
        unique.add(each.split('_')[0])
    if len(unique) == 1 and label == '1':
        correct_pr3_ex += 1
    if len(unique) != 1 and label == '0':
        correct_pr3_ex += 1
    total_pr3_ex += 1
    line = ' '.join(pr3_test_ex.readline().split())

    




    
