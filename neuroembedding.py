from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization, Reshape
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D
from conv3dTranspose import Conv3DTranspose
from keras.layers import concatenate, add, multiply
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

# from keras.utils.visualize_util import plot
from keras.callbacks import Callback
from keras import backend as K

import tensorflow as tf


def encoder():

    filters = 16
    conv_size = (3, 3, 3)
    pool_size = (2, 2, 2)

    inputs = Input()

    conv1 = Conv3D(1*filters, conv_size, activation='relu', padding='same')(inputs)
    conv1 = Conv3D(1*filters, conv_size, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size)(conv1)
    conv2 = Conv3D(2*filters, conv_size, activation='relu', padding='same')(pool1)
    conv2 = Conv3D(2*filters, conv_size, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size)(conv2)
    conv3 = Conv3D(3*filters, conv_size, activation='relu', padding='same')(pool2)
    conv3 = Conv3D(3*filters, conv_size, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size)(conv3)
    latent = Conv3D(128, conv_size, activation='relu', padding='same')(pool3)

    return Model(inputs=[inputs], outputs=[latent])

def decoder():

    filters = 16
    conv_size = (3, 3, 3)
    pool_size = (2, 2, 2)

    n_tissues = 3

    inputs = Input()

    conv1 = Conv3DTranspose(3*filters, conv_size, strides=(2, 2, 2), activation='relu', padding='same')(inputs)
    conv2 = Conv3DTranspose(3*filters, conv_size, strides=(2, 2, 2), activation='relu', padding='same')(conv1)
    conv3 = Conv3DTranspose(3*filters, conv_size, strides=(2, 2, 2), activation='relu', padding='same')(conv2)

    conv4 = Conv3D(n_tissues, (1, 1, 1), activation='relu', padding='same')(conv3)

    return Model(inputs=[inputs], outputs=[conv4])

