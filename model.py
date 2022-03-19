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
import tensorflow_io as tfio
import keras

IMG_WIDTH = 256
IMG_HEIGHT = 256
# The facade training set consist of 400 images
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 90


#HELPER FUNCTIONS=======================================================================================================================
def load(image):
  # Read and decode an image file to a uint8 tensor
  image=image/256
  image = tfio.experimental.color.rgb_to_lab(image)
  batch = tf.shape(image)[0]
  coloured = tf.image.resize(image[:,:,:,1:],(IMG_WIDTH,IMG_HEIGHT))

  grayscale = image[:,:,:,0]
  grayscale=tf.reshape(grayscale,(batch,IMG_WIDTH,IMG_HEIGHT,1))
  grayscale = tf.image.resize(grayscale,(IMG_WIDTH,IMG_HEIGHT))
  
  # Convert both images to float32 tensors
  grayscale = tf.cast(grayscale, tf.float32)
  coloured = tf.cast(coloured, tf.float32)
  return grayscale, coloured  

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image  

# Normalizing the images to [0, 1]
def normalize(input_image, real_image):
  # input_image = (input_image / 127.5) - 1
  # real_image = (real_image / 127.5) - 1
  input_image = input_image/100
  real_image = real_image/127

  return input_image, real_image

@tf.function
def random_crop(input_image, real_image):
  input_image = tf.concat([input_image, input_image], axis=-1)
  batch = tf.shape(input_image)[0]
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2,batch, IMG_HEIGHT, IMG_WIDTH,2])
  input = cropped_image[0][:,:,:,0]
  input = tf.reshape(input,[tf.shape(input)[0],tf.shape(input)[1],tf.shape(input)[2],1])
  target = cropped_image[1]
  print(input.shape)
  print(target.shape)

  return input, target

@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)
  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)
  import random
  if random.uniform(0, 1)> 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image
   


# Normalizing the images to [0, 1]
def normalize(input_image, real_image):
  # input_image = (input_image / 127.5) - 1
  # real_image = (real_image / 127.5) - 1
  input_image = input_image/100
  real_image = real_image/127

  return input_image, real_image


def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  # input_image, real_image = resize(input_image, real_image,
  #                                  IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image  


def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


#PIPELINE=======================================================================================================================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "../input/data_256",
    label_mode=None,
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

