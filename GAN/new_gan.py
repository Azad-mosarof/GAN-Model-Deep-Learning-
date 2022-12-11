from calendar import c
from pickletools import optimize
from pyexpat import model
from statistics import mode
from turtle import shape, update
from unicodedata import name
import keras 
import os
import tensorflow as tf
import numpy as np
from keras.layers import Input,Dense,Reshape,Flatten,Activation,MaxPool2D
from keras.layers import BatchNormalization,Conv2DTranspose,Conv2D,Dropout
from keras.layers import Input, ReLU
from keras.models import Sequential,Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.layers import LeakyReLU
import cv2
from keras.datasets import mnist

img_rows = 128
img_cols = 128
img_channel = 3
img_shape = (img_rows,img_cols,img_channel)
img_dir = '/home/azadm/Desktop/Datasetf_For_ML/train/train/'
file_name = "test_img/test.png"

optimizer = Adam(0.001, 0.5) # lr and momentum

noise_shape = (100,) #generator takes 1D array of size 100 as Input

input_layer = Input(shape=noise_shape, name='generator_input')
L1 = Dense(np.prod(img_shape), name="Dense_1")(input_layer)

reshape_layer = Reshape(img_shape, name="Reshape_layer_1")(L1)

ct_1 = Conv2DTranspose(256, kernel_size=(2,2), 
                    padding="same", 
                    strides=(1, 1), 
                    name="Transpose_layer_1")(reshape_layer)
ct_1 = ReLU(name="relu_1")(ct_1)
ct_1 = BatchNormalization(name="Batch_normalization_1")(ct_1)              

ct_1 = Conv2D(256, kernel_size=(2,2), 
                    padding="same", 
                    strides=(1, 1), 
                    name="conv_layer_1")(ct_1)
ct_1 = ReLU(name="relu_2")(ct_1)
ct_1 = BatchNormalization(name="Batch_normalization_2")(ct_1) 


ct_1 = Conv2DTranspose(128, kernel_size=(2,2), 
                    padding="same", 
                    strides=(1, 1), 
                    name="Transpose_layer_2")(ct_1)
ct_1 = ReLU(name="relu_3")(ct_1)
ct_1 = BatchNormalization(name="Batch_normalization_3")(ct_1)              

ct_1 = Conv2D(128, kernel_size=(2,2), 
                    padding="same", 
                    strides=(1, 1), 
                    name="conv_layer_2")(ct_1)
ct_1 = ReLU(name="relu_4")(ct_1)
ct_1 = BatchNormalization(name="Batch_normalization_4")(ct_1) 


ct_1 = Conv2DTranspose(64, kernel_size=(2,2), 
                    padding="same", 
                    strides=(1, 1), 
                    name="Transpose_layer_3")(ct_1)
ct_1 = ReLU(name="relu_5")(ct_1)
ct_1 = BatchNormalization(name="Batch_normalization_5")(ct_1)              

ct_1 = Conv2D(64, kernel_size=(2,2), 
                    padding="same", 
                    strides=(1, 1), 
                    name="conv_layer_3")(ct_1)
ct_1 = ReLU(name="relu_6")(ct_1)
ct_1 = BatchNormalization(name="Batch_normalization_6")(ct_1) 


ct_1 = Conv2DTranspose(32, kernel_size=(2,2), 
                    padding="same", 
                    strides=(1, 1), 
                    name="Transpose_layer_4")(ct_1)
ct_1 = ReLU(name="relu_7")(ct_1)
ct_1 = BatchNormalization(name="Batch_normalization_7")(ct_1)              

ct_1 = Conv2D(32, kernel_size=(2,2), 
                    padding="same", 
                    strides=(1, 1), 
                    name="conv_layer_4")(ct_1)
ct_1 = ReLU(name="relu_8")(ct_1)
ct_1 = BatchNormalization(name="Batch_normalization_8")(ct_1) 


ct_1 = Conv2DTranspose(16, kernel_size=(2,2), 
                    padding="same", 
                    strides=(1, 1), 
                    name="Transpose_layer_5")(ct_1)
ct_1 = ReLU(name="relu_9")(ct_1)
ct_1 = BatchNormalization(name="Batch_normalization_9")(ct_1)              

ct_1 = Conv2D(16, kernel_size=(2,2), 
                    padding="same", 
                    strides=(1, 1), 
                    name="conv_layer_5")(ct_1)
ct_1 = ReLU(name="relu_10")(ct_1)
ct_1 = BatchNormalization(name="Batch_normalization_10")(ct_1) 


ct_1 = Conv2DTranspose(3, kernel_size=(2,2), 
                    padding="same", 
                    strides=(1, 1), 
                    name="Transpose_layer_6")(ct_1)
ct_1 = ReLU(name="relu_11")(ct_1)
ct_1 = BatchNormalization(name="Batch_normalization_11")(ct_1)              

ct_1 = Conv2D(3, kernel_size=(2,2), 
                    padding="same", 
                    strides=(1, 1), 
                    name="conv_layer_6")(ct_1)
ct_1 = Activation("sigmoid", name="sigmoid_activation_layer")(ct_1)

generator_model = Model(input_layer, ct_1)

generator_model.summary()

generator_model.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=['accuracy'])


## Build discriminator model

input_layer  = Input(shape=(img_shape), name="Discriminator_input_layer")

c_1 = Conv2D(256, (2,2), 
            padding="same", 
            strides=(2,2), 
            name="d_conv_1",
            activation='relu')(input_layer)
c_1 = BatchNormalization(name="d_Batch_normalization_1")(c_1)

c_1 = Conv2D(128, (2,2), 
            padding="same", 
            strides=(2,2), 
            name="d_conv_2",
            activation='relu')(c_1)
c_1 = BatchNormalization(name="d_Batch_normalization_2")(c_1)

c_1 = Conv2D(64, (2,2), 
            padding="same", 
            strides=(2,2), 
            name="d_conv_3",
            activation='relu')(c_1)
c_1 = BatchNormalization(name="d_Batch_normalization_3")(c_1)

c_1 = Conv2D(32, (2,2), 
            padding="same", 
            strides=(2,2), 
            name="d_conv_4",
            activation='relu')(c_1)
c_1 = BatchNormalization(name="d_Batch_normalization_4")(c_1)

c_1 = Conv2D(16, (2,2), 
            padding="same", 
            strides=(2,2), 
            name="d_conv_5",
            activation='relu')(c_1)
c_1 = BatchNormalization(name="d_Batch_normalization_5")(c_1)

c_1 = Conv2D(8, (2,2), 
            padding="same", 
            strides=(2,2), 
            name="d_conv_6",
            activation='relu')(c_1)
c_1 = BatchNormalization(name="d_Batch_normalization_6")(c_1)


c_1 = Flatten(name="Flatten_layer_1")(c_1)

c_1 = Dense(1, activation='sigmoid')(c_1)

discriminator_model = Model(input_layer, c_1)

discriminator_model.summary()

discriminator_model.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=['accuracy'])

def GAN(generator,discriminator):
    discriminator.trainable = False
    # Connet generator and discriminator
    model = Sequential(name="GAN")
    model.add(generator)
    model.add(discriminator)
    return model

gan = GAN(generator_model,discriminator_model)

gan.summary()

gan.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=['accuracy'])

