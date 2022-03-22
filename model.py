#IMPORTS=======================================================================================================================
import tensorflow as tf
import os
import pathlib
import time
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Activation,Reshape, UpSampling2D,Conv2DTranspose
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from skimage.io import imshow
from skimage.color import rgb2lab, lab2rgb
from IPython import display
import pathlib
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import optimizers
# import tensorflow_io as tfio
from helper import *
import keras

# The facade training set consist of 400 images
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 90





#PIPELINE=======================================================================================================================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "../../input/data_256/",
    label_mode=None,
    labels='inferred',
    batch_size=BATCH_SIZE)

train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)


val_dataset = tf.keras.utils.image_dataset_from_directory(
    "../input/val",
    label_mode=None,
    batch_size=BATCH_SIZE)

val_dataset = val_dataset.map(load_image_test)


test_dataset = tf.keras.utils.image_dataset_from_directory(
    "../input/test",
    label_mode=None,
    batch_size=BATCH_SIZE)

test_dataset = test_dataset.map(load_image_test)



#MODEL==================================================================================================
#Construct the model
model = Sequential()

#Encoder
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, input_shape=(256, 256, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))

#Decoder
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))


#Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mse' , metrics=['accuracy'])





#TRAIN===========================================================================================
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history=model.fit(train_dataset, epochs = 20,shuffle=True,callbacks=[model_checkpoint_callback], validation_data=val_dataset)



#SAVE==========================================================================================
model.save("model.h5")
np.save("history.npy",history.history)

