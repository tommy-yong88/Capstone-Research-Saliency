# -*- coding: utf-8 -*-
import os, cv2, sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Dropout, Conv2D, Lambda,BatchNormalization, Activation, Add, Input, MaxPooling2D, AveragePooling2D, Flatten, GlobalMaxPooling2D, ZeroPadding2D, Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import tensorflow.keras.backend as K
import math
import scipy.io
import scipy.ndimage
from model import TPNet
from utils import kld, getTrainData, getValData, cross_val_split_data, generatorTrain, generatorVal, padding, preprocess_images, preprocess_maps, postprocess_predictions, loss, TVdist, plotLost

### Constants
kfolds = 5
epochs = 20
batch_size = 5
checkpoint_filepath = './'
# path of training images
imgs_train_path = './images/train/images/'
# path of training maps
maps_train_path = './images/train/maps/'
# number of training images
nb_imgs_train = 10000
# path of validation images
imgs_val_path = './images/validation/images/'
# path of validation maps
maps_val_path = './images/validation/maps/'
# number of validation images
nb_imgs_val = 5000
    
### Main 
model = TPNet()
model.summary()
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss=TVdist)

# Get all training and validation data
trainImages, trainMaps = getTrainData()
valImages, valMaps = getValData()

model_earlystop_callback = EarlyStopping(patience=20)

kfoldTrainImages, kfoldTrainMaps, kfoldValImages, kfoldValMaps = cross_val_split_data(trainImages, trainMaps, valImages, valMaps)
csv_logger = CSVLogger('training.log')
    
history = model.fit(generatorTrain(batch_size, kfoldTrainImages, kfoldTrainMaps), 
              epochs=epochs,verbose = 1, steps_per_epoch=nb_imgs_train,
              callbacks=[ModelCheckpoint(
                  filepath="weights_{epoch:02d}-{val_loss:.4f}.h5",
                  save_best_only=False),model_earlystop_callback],
                  validation_data=generatorVal(batch_size, kfoldValImages, kfoldTrainMaps),
                  validation_steps=nb_imgs_val)
    
plotLost(history)