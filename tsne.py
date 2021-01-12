# -*- coding: utf-8 -*-
import os, cv2, sys
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import pandas as pd
import time
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from PIL import Image
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import seaborn as sns
import matplotlib.patches as mpatches

shape_r = 480
shape_c = 640

model = TPNet()
model.summary()

model = Model(inputs=model.inputs, outputs=model.layers[17].output)
model.summary()
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_images(['img13.jpg'], shape_r, shape_c)
img2 = preprocess_images(['img51.jpg'], shape_r, shape_c)

# get feature map for first hidden layer
feature_maps1 = model.predict(img)
feature_maps2 = model.predict(img2)
feature_map_final = []

for i in range (0, feature_maps1.shape[-1]):
    feature_map_final.append(feature_maps1[0,:,:,i])
for i in range (0, feature_maps2.shape[-1]):
    feature_map_final.append(feature_maps2[0,:,:,i])

flattened_feature_map1 = []
for i in range (0, feature_maps1.shape[-1]):
    fmap = np.array(feature_maps1[0,:,:,i])
    fmap = fmap.flatten()
    flattened_feature_map1.append(fmap)

flattened_feature_map1 = normalize(flattened_feature_map1[:])

feat_cols = [ 'pixel'+str(i) for i in range(flattened_feature_map1.shape[1]) ]

df = pd.DataFrame(flattened_feature_map1, columns=feat_cols)
df['y'] = 1

flattened_feature_map2 = []
for i in range (0, feature_maps2.shape[-1]):
    fmap = np.array(feature_maps2[0,:,:,i])
    fmap = fmap.flatten()
    flattened_feature_map2.append(fmap)

flattened_feature_map2 = normalize(flattened_feature_map2[:])

df2 = pd.DataFrame(flattened_feature_map2, columns=feat_cols)
df2['y'] = 2

df = df.append(df2, ignore_index=True)
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=3, n_iter=300)
tsne_results = tsne.fit_transform(df)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two'])
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
df_subset['y']= df['y'] 
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="full",
    alpha=0.3
)

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

pixel_cols = df_subset.columns.str.startswith('pixel')
img_w, img_h = 28,28
zoom = 0.5
color_dict = {
  "1.0": "red",
  "2.0": "blue"
}

fig, ax = plt.subplots(figsize=(16,10))
for i,row in df_subset.iterrows():
    image = Image.fromarray(feature_map_final[i]).resize((30,40))
    im = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(im, (row["tsne-2d-one"], row["tsne-2d-two"]), xycoords='data', bboxprops =dict(edgecolor=color_dict[str(row['y'])]))
    ax.add_artist(ab)
    ax.update_datalim([(row["tsne-2d-one"], row["tsne-2d-two"])])
    ax.autoscale()
red_patch = mpatches.Patch(color='red', label='Default')
blue_patch = mpatches.Patch(color='blue', label='Enhanced')
ax.legend(handles=[red_patch, blue_patch])
ax.set_xlabel('tsne-2d-one')
ax.set_ylabel('tsne-2d-two')
