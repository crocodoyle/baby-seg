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

    enc_conv1a = Conv3D(1*filters, conv_size, activation='relu', padding='same')(inputs)
    enc_conv1b = Conv3D(1*filters, conv_size, activation='relu', padding='same')(enc_conv1a)
    enc_pool1 = MaxPooling3D(pool_size)(enc_conv1b)
    enc_conv2a = Conv3D(2*filters, conv_size, activation='relu', padding='same')(enc_pool1)
    enc_conv2b = Conv3D(2*filters, conv_size, activation='relu', padding='same')(enc_conv2a)
    enc_pool2 = MaxPooling3D(pool_size)(enc_conv2b)
    enc_conv3a = Conv3D(3*filters, conv_size, activation='relu', padding='same')(enc_pool2)
    enc_conv3b = Conv3D(3*filters, conv_size, activation='relu', padding='same')(enc_conv3a)
    enc_pool3 = MaxPooling3D(pool_size)(enc_conv3b)
    enc_conv4a = Conv3D(4*filters, conv_size, activation='relu', padding='same')(enc_pool3)
    enc_conv4b = Conv3D(4*filters, conv_size, activation='relu', padding='same')(enc_conv4a)
    encoded = MaxPooling3D(pool_size)(enc_conv4b)

    # enc_flat = Flatten()(enc_pool3)
    # latent = Conv3D(128, conv_size, activation='relu', padding='same')(enc_pool3)
    # latent = Dense(128)(enc_flat)

    return Model(inputs=[inputs], outputs=[encoded])


def decoder():
    filters = 16
    conv_size = (3, 3, 3)
    pool_size = (2, 2, 2)

    n_tissues = 3

    encoded_shape = (9, 12, 8, 1)

    encoded_inputs = Input(shape=encoded_shape)

    dec_conv1 = Conv3DTranspose(4*filters, conv_size, strides=pool_size, activation='relu', padding='same')(encoded_inputs)
    dec_conv2 = Conv3DTranspose(3*filters, conv_size, strides=pool_size, activation='relu', padding='same')(dec_conv1)
    dec_conv3 = Conv3DTranspose(2*filters, conv_size, strides=pool_size, activation='relu', padding='same')(dec_conv2)
    dec_conv4 = Conv3DTranspose(1*filters, conv_size, strides=pool_size, activation='relu', padding='same')(dec_conv3)
    decoded = Conv3D(n_tissues, (1, 1, 1), activation='relu', padding='same')(dec_conv4)

    return Model(inputs=[encoded_inputs], outputs=[decoded])


def autoencoder():
    label_inputs = Input(shape=(144, 192, 128, 3))

    enc = encoder()(label_inputs)
    dec = decoder()(enc)

    outputs = dec

    autoenc = Model(inputs=[label_inputs], outputs=[outputs])

    return autoenc


def t_net():
    mri_inputs = Input(shape=(144, 192, 128, 2))

    decod = decoder()
    decod.trainable = False

    conv = convnet()(mri_inputs)

    decoded = decod(conv)

    return Model(inputs=[mri_inputs], outputs=[decoded])


def convnet():
    conv_size = (3, 3, 3)
    filters = 4

    mri_inputs = Input(shape=(144, 192, 128, 2))

    conv1 = Conv3D(filters*1, conv_size, activation='relu', padding='valid')(mri_inputs)
    drop1 = Dropout(0.5)(conv1)
    conv2 = Conv3D(filters*2, conv_size, activation='relu', padding='valid')(drop1)
    drop2 = Dropout(0.5)(conv2)
    conv3 = Conv3D(filters*4, conv_size, activation='relu', padding='valid')(drop2)
    drop3 = Dropout(0.5)(conv3)
    conv4 = Conv3D(filters*8, conv_size, activation='relu', padding='valid')(drop3)
    drop4 = Dropout(0.5)(conv4)
    flat = Flatten()(drop4)

    latent = Dense(64)(flat)

    return Model(inputs=[mri_inputs], outputs=[latent])


def tl_net():
    label_inputs = Input(shape=(144, 192, 128, 3))
    mri_inputs = Input(shape=(144, 192, 128, 2))

    conv = convnet()(mri_inputs)
    encoded = encoder()(label_inputs)

    mul = multiply([encoded, conv])

    decoded = decoder()(mul)

    return Model(inputs=[label_inputs, mri_inputs], outputs=[decoded])
