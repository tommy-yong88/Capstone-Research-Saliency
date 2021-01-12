# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 22:25:27 2020

@author: Tommy
"""
import os, cv2, sys
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import h5py
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
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
from model import TPNet
from utils import getTestData, getValData, cross_val_split_data, generatorTrain, generatorVal, padding, preprocess_images, preprocess_maps, postprocess_predictions, loss, TVdist, plotLost

img_test_path = './test/images/'
output_folder = './test/prediction/'
shape_r = 480
shape_c = 640

### Main 
model = TPNet()
model.summary()

# summarize filter shapes
for layer in model.layers:
	# check for convolutional layer
	if 'Conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)
    
# retrieve weights from the second hidden layer
filters, biases = model.layers[19].get_weights()

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot first few filters
n_filters, ix = 20, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
pyplot.show()

#18 is final maxpooling layer

model = Model(inputs=model.inputs, outputs=model.layers[19].output)
model.summary()
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_images(['img13.jpg'], shape_r, shape_c)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot all 64 maps in an 8x8 squares
row = 8
column = 8
ix = 1

if feature_maps.shape[-1] > 1:
    for _ in range(row):
    	for _ in range(column):
    		# specify subplot and turn of axis
    		ax = pyplot.subplot(row, column, ix)
    		ax.set_xticks([])
    		ax.set_yticks([])
    		# plot filter channel in grayscale
    		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
    		ix += 1
    pyplot.show()
else:
    pyplot.imshow(feature_maps[0, :, :, 0], cmap='gray')