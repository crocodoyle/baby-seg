from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization, Reshape
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D
from keras.layers import concatenate, add, multiply
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

# from keras.utils.visualize_util import plot
from keras.callbacks import Callback
from keras import backend as K

import tensorflow as tf


def encoder(inputs):

    filters = 16
    conv_size = (3, 3, 3)
    pool_size = (2, 2, 2)

    inputs = inputs

    conv1 = Conv3D(1*filters, conv_size, activation='relu', padding='same')(inputs)
    conv1 = Conv3D(1*filters, conv_size, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size)(conv1)
    conv2 = Conv3D(2*filters, conv_size, activation='relu', padding='same')(pool1)
    conv2 = Conv3D(2*filters, conv_size, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size)(conv2)
    conv3 = Conv3D(3*filters, conv_size, activation='relu', padding='same')(pool2)
    conv3 = Conv3D(3*filters, conv_size, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size)(conv3)
    conv4 = Conv3D(64, (1, 1, 1), activation='relu', padding='same')(pool3)

def decoder(inputs):

    filters = 16
    conv_size = (3, 3, 3)
    pool_size = (2, 2, 2)

    conv1 = Conv3D(1*filters, conv_size, activation='relu', padding='same')(inputs)
    conv1 = Conv3D(1*filters, conv_size, activation='relu', padding='same')(conv1)
    up1 = UpSampling3D(pool_size)(conv1)
    conv2 = Conv3D(2*filters, conv_size, activation='relu', padding='same')(up1)
    conv2 = Conv3D(2*filters, conv_size, activation='relu', padding='same')(conv2)
    up2 = UpSampling3D(pool_size)(conv2)
    conv3 = Conv3D(3*filters, conv_size, activation='relu', padding='same')(up2)
    conv3 = Conv3D(3*filters, conv_size, activation='relu', padding='same')(conv3)
    pool3 = UpSampling3D(pool_size)(conv3)

    conv4 = Conv3D(64, (1, 1, 1), activation='relu', padding='same')(pool3)