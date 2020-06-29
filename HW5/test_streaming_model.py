# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 16:50:22 2020

@author: Lenovo
"""

def load_stream_model():
    streaming_in_shape = train_data.shape[2:]
    streaming_in = Input(batch_shape=(1,None,streaming_in_shape[0]))  ## stateful ==> needs batch_shape specified
    x = Dense(128, activation='relu')(streaming_in)
    x = GRU(256, 
            return_sequences=True, 
            stateful=True, 
            kernel_regularizer=regularizers.l2(0.001),
            recurrent_regularizer=regularizers.l2(0.001),
            dropout=0.4)(x)
    x = BatchNormalization()(x)
    x = GRU(256, 
            return_sequences=False, 
            stateful=True, 
            kernel_regularizer=regularizers.l2(0.001),
            recurrent_regularizer=regularizers.l2(0.001),
            dropout=0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    streaming_out = Dense(3, activation='softmax')(x)
    streaming_model = Model(inputs=streaming_in, outputs=streaming_out)
    streaming_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    streaming_model.summary()
    streaming_model.load_weights('weights_0_30_silence_balance.hd5')
    return streaming_model