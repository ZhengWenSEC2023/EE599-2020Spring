from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, add, BatchNormalization, concatenate, Flatten
from tensorflow.keras.layers import Conv2D, Input, Dropout, MaxPooling2D
from tensorflow.keras.models import Model
from data_pr3 import polyvore_dataset, DataGenerator, DataGenerator_test
from utils_pr3 import Config
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import h5py
from tensorflow.keras.utils import plot_model
import numpy as np


class OnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback()
      


if __name__=='__main__':

    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_val, y_train, y_val = dataset.create_dataset_train()
    X_test = dataset.create_dataset_test()
    
    set_train = []
    for i in range(len(X_train)):
        set_train.append([X_train[i], y_train[i]])

    set_train = set_train[:Config['num_train']//2] + set_train[-Config['num_train']//2:]
    np.random.shuffle(set_train)
    
    X_train = [];
    y_train = [];
    for i in range(len(set_train)):
        X_train.append(set_train[i][0])
        y_train.append(set_train[i][1])
    
    
    set_val = []
    for i in range(len(X_val)):
        set_val.append([X_val[i], y_val[i]])
    set_val = set_val[:Config['num_val']//2] + set_val[-Config['num_val']//2:]
    np.random.shuffle(set_val)
    
    X_val = [];
    y_val = [];
    for i in range(len(set_val)):
        X_val.append(set_val[i][0])
        y_val.append(set_val[i][1])
        
    
    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        val_set = (X_val[:100], y_val[:100], transforms['val'])
        test_set = (X_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'val': 100, 'test': 100}
    else:
        train_set = (X_train[:Config['num_train']], y_train[:Config['num_train']], transforms['train'])
        val_set = (X_val[:Config['num_val']], y_val[:Config['num_val']], transforms['val'])
        test_set = (X_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'val': len(y_val), 'test': len(X_test)}

    params = {'batch_size': Config['batch_size'],
              'shuffle': True}

    train_generator =  DataGenerator(train_set, dataset_size, params)
    val_generator = DataGenerator(val_set, dataset_size, params)
    test_generator = DataGenerator_test(test_set, dataset_size, params)

    # Use GPU

    
        # build model        
    inputShape = (224, 224, 3)
    chanDim = -1
    
    # input block 1
    inputs_1 = Input(shape=inputShape, name='Input_1')
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(inputs_1)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    output_1_1 = Dropout(0.25)(x)
    
    # input block 2
    inputs_2 = Input(shape=inputShape, name='Input_2')
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(inputs_2)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    output_2_1 = Dropout(0.25)(x)
    
    output_2 = concatenate([output_1_1, output_2_1])
    
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(output_2)
    x = BatchNormalization(axis=chanDim)(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    output_2 = GlobalAveragePooling2D()(x)
    
    x = Dense(256, activation='relu')(output_2)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    predictions = Dense(2, activation = 'softmax')(x)
    model = Model(inputs=[inputs_1, inputs_2], outputs=predictions)

    # define optimizers
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    # plot_model(model, to_file='my_test_model_Pr3.jpg', show_shapes=True)
    # training - num worker is obsolete now
    
    
    def scheduler(epoch):
        begin_epoch = 5
        rate = Config['learning_rate']
        if epoch < begin_epoch:
            return rate
        else:
            return rate * np.exp(0.2 * (begin_epoch - epoch))
    
    learning_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)
    check_point = tf.keras.callbacks.ModelCheckpoint('model_check.hdf5', monitor='val_loss', verbose=1)
    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=Config['num_epochs'],
                        callbacks=[learning_rate, check_point, OnEpochEnd([train_generator.on_epoch_end])]
                        )
    train_eval = model.evaluate(train_generator)
    val_eval = model.evaluate(val_generator)
    
    predictions = model.predict(test_generator)
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    model.save('my_test_model_Pr3.hdf5')
    f = h5py.File("prob_3_test_1.hdf5", "w")
    f['loss'] = loss
    f['val_loss'] = val_loss
    f['accuracy'] = acc
    f['val_accuracy'] = val_acc
    f['predictions'] = predictions
    f['train_eval'] = train_eval
    f['val_eval'] = val_eval
    f.close()
    # plot_model(model, to_file='/home/ubuntu/EE599-CV-Project/keras/Pr_3/Pr3.pdf', show_shapes=True)

    