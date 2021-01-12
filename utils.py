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

### Constants
img_rows = 480
img_cols = 640
shape_r = 480
shape_c = 640

imgs_train_path = './images/train/images/'
maps_train_path = './images/train/maps/'
imgs_val_path = './images/validation/images/'
maps_val_path = './images/validation/maps/'
img_test_path = './test/images/'
output_folder = './test/prediction/'

def getTrainData():    
    images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith('.jpg')]
    maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith('.png')]
    
    images.sort()
    maps.sort()
        
    return images, maps
    
def getValData(): 
    images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith('.jpg')]
    maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith('.png')]
    
    images.sort()
    maps.sort()
    
    return images, maps    

def cross_val_split_data(kfolds, nfold,trainImage, trainMaps, valImages, valMaps):
    splitTrainImages = []
    splitTrainMaps = []
    splitValImages = []
    splitValMaps = []
    
    totalTrainImages = len(trainImage)
    imagesPerPart = int(totalTrainImages/kfolds)

    for i in range(0, kfolds):
        start = int((i*imagesPerPart))
        end = int((i*imagesPerPart) + imagesPerPart)

        trainImages = trainImage[start:end]
        
        if i == nfold:
            splitValImages = valImages + trainImages
            splitValMaps = valMaps + trainMaps[start:end]
        else:
            splitTrainImages = splitTrainImages + trainImages
            splitTrainMaps = splitTrainMaps + trainMaps[start:end]
    
    return splitTrainImages, splitTrainMaps, splitValImages, splitValMaps
    
# MLNET sample start
def generatorTrain(b_s, images, maps):
    counter = 0
    while True:
        yield preprocess_images(images[counter:counter + b_s], shape_r, shape_c), preprocess_maps(maps[counter:counter + b_s], shape_r, shape_c)
        counter = (counter + b_s) % len(images)
        
def generatorVal(b_s, images, maps):
    counter = 0
    while True:
        yield preprocess_images(images[counter:counter + b_s], shape_r, shape_c), preprocess_maps(maps[counter:counter + b_s], shape_r, shape_c)
        counter = (counter + b_s) % len(images)
        
def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def preprocess_images(paths, shape_r, shape_c,mirror=False):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))

    for i, path in enumerate(paths):
        original_image = cv2.imread(path)
        padded_image = padding(original_image, shape_r, shape_c, 3)
        if mirror:
            flip = - 1
            padded_image = padded_image[:, :, ::flip]
        ims[i] = padded_image

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68

    return ims


def preprocess_maps(paths, shape_r, shape_c,mirror=False):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        original_map = cv2.imread(path, 0)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        if mirror:
            flip = - 1
            padded_map = padded_map[:, ::flip]
        ims[i,:,:,0] = padded_map.astype(np.float32)
        ims[i,:,:,0] /= 255.0

    return ims


def postprocess_predictions(pred, shape_r, shape_c,sigma=19):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    img = scipy.ndimage.filters.gaussian_filter(img, sigma)
    img -= np.min(img)
    img = img /np.max(img)
    
    return img * 255

def getTestData(imgs_test_path):    
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith('.jpg')]
    images.sort()
        
    return images

def loss(y_true, y_pred):
    max_y = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), shape_r, axis=0)), shape_c, axis=0)
    return K.mean(K.square((y_pred / max_y) - y_true) / (1 - y_true + 0.1))

def kld(y_true, y_pred, eps=1e-7):
    """This function computes the Kullback-Leibler divergence between ground
       truth saliency maps and their predictions. Values are first divided by
       their sum for each image to yield a distribution that adds to 1.

    Args:
        y_true (tensor, float32): A 4d tensor that holds the ground truth
                                  saliency maps with values between 0 and 255.
        y_pred (tensor, float32): A 4d tensor that holds the predicted saliency
                                  maps with values between 0 and 1.
        eps (scalar, float, optional): A small factor to avoid numerical
                                       instabilities. Defaults to 1e-7.

    Returns:
        tensor, float32: A 0D tensor that holds the averaged error.
    """

    sum_per_image = tf.reduce_sum(y_true, axis=(1, 2, 3), keepdims=True)
    y_true /= eps + sum_per_image

    sum_per_image = tf.reduce_sum(y_pred, axis=(1, 2, 3), keepdims=True)
    y_pred /= eps + sum_per_image

    loss = y_true * tf.math.log(eps + y_true / (eps + y_pred))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=(1, 2, 3)))

    return loss

def TVdist(y_true, y_pred):
        P = y_true
        P = P / (K.epsilon() + K.sum(P, axis=[1, 2, 3], keepdims=True))
        Q = y_pred
        Q = Q / (K.epsilon() + K.sum(Q, axis=[1, 2, 3], keepdims=True))

        tv = K.sum( K.abs(P - Q) , axis=[1, 2, 3])
        return tv*0.5

def plotLost(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    plt.plot(epochs, loss, label = 'Training Loss')
    plt.plot(epochs, val_loss, label = 'Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
def generatorTest(images):
    counter = 0
    b_s=1
    while True:
        yield preprocess_images(images[counter:counter + b_s], shape_r, shape_c)
        counter = (counter + b_s) % len(images)   
