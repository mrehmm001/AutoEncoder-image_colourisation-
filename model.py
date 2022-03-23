#IMPORTS=======================================================================================================================
import tensorflow as tf
import os
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
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import optimizers
# import tensorflow_io as tfio
from helper import *

# The facade training set consist of 400 images
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 90





#PIPELINE=======================================================================================================================
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    "../../input/data_256/",
    batch_size=BATCH_SIZE)

train_dataset = preprocessor(train_generator,load_image_train)

val_datagen = ImageDataGenerator()
val_generator = val_dataset.flow_from_directory(
    "../input/val",
    batch_size=BATCH_SIZE)

val_dataset = preprocessor(val_generator,load_image_test)


test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    "../input/test",
    batch_size=BATCH_SIZE)

test_dataset = preprocessor(test_generator,load_image_test)



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

