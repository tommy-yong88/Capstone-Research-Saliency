# -*- coding: utf-8 -*-
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import UpSampling2D, concatenate, Dropout, Conv2D, Lambda,BatchNormalization, Activation, Add, Input, MaxPooling2D, AveragePooling2D, Flatten, GlobalMaxPooling2D, ZeroPadding2D, Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import tensorflow.keras.backend as K
import math
import scipy.io
import scipy.ndimage

### Constants
kfolds = 5
img_rows = 480
img_cols = 640

def TPNet():
    #Input
    X_input = Input(shape=(img_rows, img_cols,3))
    
    Y = AveragePooling2D()(X_input)
    
    X = Conv2D(64, 3, activation='relu',padding='same', name = "x_Conv1",)(X_input)
    X = Conv2D(64, 3, activation='relu',padding='same', name = "x_Conv2",)(X)
    X = MaxPooling2D(pool_size=(2,2), strides = (2,2))(X)
    
    X = Conv2D(128, 3, activation='relu',padding='same', name = "x_Conv3",)(X)
    X = Conv2D(128, 3, activation='relu',padding='same', name = "x_Conv4",)(X)
    X = MaxPooling2D(pool_size=(2,2), strides = (2,2))(X)
    
    X = Conv2D(256, 3, activation='relu',padding='same', name = "x_Conv5",)(X)
    X = Conv2D(256, 3, activation='relu',padding='same', name = "x_Conv6",)(X)
    X = Conv2D(256, 3, activation='relu',padding='same', name = "x_Conv7",)(X)
    X = MaxPooling2D(pool_size=(2,2), strides = (2,2))(X)
    
    X = Conv2D(512, 3, activation='relu',padding='same', name = "x_Conv8",)(X)
    X = Conv2D(512, 3, activation='relu',padding='same', name = "x_Conv9",)(X)
    X = Conv2D(512, 3, activation='relu',padding='same', name = "x_Conv10",)(X)    
    X = MaxPooling2D(pool_size=(2,2), strides = (2,2))(X)

    X = Conv2D(512, 3, activation='relu',padding='same', name = "x_Conv11",)(X)
    X = Conv2D(512, 3, activation='relu',padding='same', name = "x_Conv12",)(X)
    X = Conv2D(512, 3, activation='relu',padding='same', name = "x_Conv13",)(X)    
    X = MaxPooling2D(pool_size=(2,2), strides = (2,2))(X)
    
    #final = concatenate([X, Y], axis = 3)
            
    final_output = Conv2D(1, kernel_size=(1, 1), activation='relu',padding='same')(X)
    
    final_output_up = Lambda(lambda X_1: tf.compat.v1.image.resize_bilinear(X_1, size=(480, 640)))(final_output)  

    model = Model(inputs = X_input, outputs = final_output_up, name = 'TPNET')
    return model



def conv_block(X, f, filters, stage, block, s = 2):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    use_bias=True
    dilation_rate=(1, 1)
    
    filter1, filter2 = filters
    
    X_shortcut = X
    
    X = Conv2D(filter1, 3, strides = s, padding = 'same', use_bias=use_bias,name = conv_name_base + '2a', kernel_initializer = 'glorot_uniform', data_format='channels_last')(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filter2, 3, strides = 1, padding = 'same', dilation_rate=dilation_rate, use_bias=use_bias, name = conv_name_base + '2b', kernel_initializer = 'glorot_uniform', data_format='channels_last')(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    X_shortcut = Conv2D(filter2, 1, strides = s, use_bias=use_bias, padding = 'same', name = conv_name_base + '1', kernel_initializer = 'glorot_uniform', data_format='channels_last')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name = bn_name_base + '1')(X_shortcut)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def iden_block(X, f, filters, stage, block):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    use_bias=True
    dilation_rate=(1, 1)
    
    filter1, filter2 = filters
    
    X_shortcut = X
    
    X = Conv2D(filter1, 3, strides = 1, use_bias=use_bias, padding = 'same', name = conv_name_base + '2a', kernel_initializer = 'glorot_uniform', data_format='channels_last')(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filter2, 3, strides = 1, dilation_rate=dilation_rate, use_bias=use_bias, padding = 'same', name = conv_name_base + '2b', kernel_initializer = 'glorot_uniform', data_format='channels_last')(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X