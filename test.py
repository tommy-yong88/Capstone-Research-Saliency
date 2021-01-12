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
from tensorflow.keras.layers import Concatenate, Conv2D, Lambda,BatchNormalization, Activation, Add, Input, MaxPooling2D, AveragePooling2D, Flatten, GlobalMaxPooling2D, ZeroPadding2D, Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import tensorflow.keras.backend as K
import math
import scipy.io
import scipy.ndimage
from tensorflow.keras.regularizers import l1, l2
from model import TPNet
from utils import getTestData, generatorTest, getTrainData, getValData, cross_val_split_data, generatorTrain, generatorVal, padding, preprocess_images, preprocess_maps, postprocess_predictions, loss, TVdist, plotLost

### Constants
batch_size = 5
checkpoint_filepath = './'
img_test_path = './test/images/'
output_folder = './test/prediction/'

### Main 
model = TPNet()
model.summary()
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss=TVdist)

# Get all training and validation data
testImages = getTestData(img_test_path)

file_names = [f for f in os.listdir(img_test_path) if f.endswith('.jpg')]
file_names.sort()
nb_imgs_test = len(file_names)

print("Loading weights")
model.load_weights('resnet_weights.h5')
print("Predict saliency maps for " + img_test_path)
predictions = model.predict(generatorTest(images=testImages), steps=70)

for pred, name in zip(predictions, file_names):
    original_image = cv2.imread(img_test_path + name, 0)
    res = postprocess_predictions(pred, original_image.shape[0], original_image.shape[1],sigma=7)
    cv2.imwrite(output_folder + '%s' % name, res.astype(int))

