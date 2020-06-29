from sklearn.preprocessing import LabelEncoder
from post_data import polyvore_dataset, DataGenerator_test
from post_utils import Config
import tensorflow as tf
import os.path as osp
import os
import numpy as np
import json


if __name__=='__main__':
    
    model_path = 'mobModel.hdf5'
    write_path = model_path + '.txt'
    image_dir = osp.join(Config['root_path'], 'images')
    
    meta_file = open(osp.join(Config['root_path'], Config['meta_file']), 'r')
    meta_json = json.load(meta_file)
    id_to_category = {}
    for k, v in meta_json.items():
        id_to_category[k] = v['category_id']
        
    files = os.listdir(image_dir)
    y = []
    for x in files:
        if x[:-4] in id_to_category:
            y.append(int(id_to_category[x[:-4]]))

    y_decoder = LabelEncoder().fit_transform(y)
    
    
    label_map_l_caID = {}
    for i in range(len(y)):
        label_map_l_caID[y_decoder[i]] = y[i]

    
    
    
    
    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_test = dataset.create_dataset_test()

    if Config['debug']:
        test_set = (X_test[:100], transforms['test'])
        dataset_size = {'test': 100}
    else:
        test_set = (X_test, transforms['test'])
        dataset_size = {'test': len(X_test)}

    params = {'batch_size': Config['batch_test'],
              'shuffle': True
              }

    test_generator = DataGenerator_test(test_set, dataset_size, params)
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(test_generator, verbose=1)
    predictions_label = np.argmax(predictions, axis=1)
    
    test_id = []
    for each in X_test:
        test_id.append(each.split('\\')[-1].split('.')[0])
    
    f = open(write_path, 'w')
    for i in range(len(test_set[0])):
        f.write(test_id[i] + ' ' + str(label_map_l_caID[ predictions_label[i]]))
        f.write('\n')
        
    f.close()