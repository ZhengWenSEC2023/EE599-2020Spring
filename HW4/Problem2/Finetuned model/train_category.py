from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from data import polyvore_dataset, DataGenerator
from utils import Config
from tensorflow.keras.callbacks import LearningRateScheduler

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

import h5py
import matplotlib.pyplot as plt
import numpy as np


class OnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback()


if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes = dataset.create_dataset()

    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        test_set = (X_test[:100], y_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = (X_train, y_train, transforms['train'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }

    train_generator =  DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)


    # Use GPU
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    
    predictions = Dense(n_classes, activation = 'softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    fine_tune_at = 150
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # define optimizers
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    
    
    def scheduler(epoch):
        begin_epoch = 5
        rate = Config['learning_rate']
        if epoch < begin_epoch:
            return rate
        else:
            return rate * np.exp(0.2 * (begin_epoch - epoch))
    
    learning_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)
    check_point = tf.keras.callbacks.ModelCheckpoint('model_check.hdf5', monitor='val_loss', verbose=1)
    # training
    # model.fit_generator(generator=train_generator,
    #                     validation_data=test_generator,
    #                     use_multiprocessing=True,
    #                     workers=Config['num_workers'],
    #                     )
    history = model.fit(train_generator,
                        validation_data=test_generator,
                        epochs=Config['num_epochs'],
                        callbacks=[learning_rate, check_point, OnEpochEnd([train_generator.on_epoch_end])]
                        )
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # ###### fine-tune
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    #     fine_tune_at = 120
    #     for layer in base_model.layers[:fine_tune_at]:
    #         layer.trainable =  False
    #     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # model.summary()    
    
    f = h5py.File("mobNetHis.hdf5", "w")
    f['loss'] = loss
    f['val_loss'] = val_loss
    f['accuracy'] = acc
    f['val_accuracy'] = val_acc
    
    model.save('mobModel.hdf5')
    
    f.close()
    
    # plot_model(model, to_file='/home/ubuntu/EE599-CV-Project/keras/Pr2_Mob/Pr2_Mob.pdf', show_shapes=True)
##############################################################################


    # f = h5py.File("resNetHis.hdf5", "r")
    # loss = f['loss']
    # val_loss = f['val_loss']
    # accuracy = f['accuracy']
    # val_accuracy = f['val_accuracy']
    # plt.figure()
    # plt.plot(loss, label='loss')
    # plt.plot(val_loss, label='val_loss')
    # plt.figure()
    # plt.plot(accuracy, label='accuracy')
    # plt.plot(val_accuracy, label='val_accuracy')
    
    # new_model = tf.keras.models.load_model('ResModel')