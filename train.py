from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, \
    SpatialDropout2D, merge
from keras.layers import Conv3D, MaxPooling3D, SpatialDropout3D, UpSampling3D
from keras.layers.merge import Concatenate
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from keras.utils.visualize_util import plot

from keras import backend as K

import numpy as np
import h5py

import os
import nibabel as nib

import pickle as pkl

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix

import argparse

scratch_dir = '/data1/data/iSeg-2017/'
input_file = scratch_dir + 'baby-seg.hdf5'

def segmentation_model():
    """
    3D U-net model, using very small convolutional kernels
    """
    concat_axis = 4
    tissue_classes = 4

    conv_size = (3, 3, 3)
    pool_size = (2, 2, 2)

    inputs = Input(shape=(144, 192, 256, 2))
    nconv = 16
    conv1 = Conv3D(nconv, conv_size, activation='relu', padding='same')(inputs)
    conv1 = Conv3D(nconv, conv_size, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)

    conv2 = Conv3D(nconv, conv_size, activation='relu', padding='same')(pool1)
    conv2 = Conv3D(nconv, conv_size, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)

    conv3 = Conv3D(nconv, conv_size, activation='relu', padding='same')(pool2)
    conv3 = Conv3D(nconv, conv_size, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)

    conv4 = Conv3D(nconv, conv_size, activation='relu', padding='same')(pool3)
    conv4 = Conv3D(nconv, conv_size, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=pool_size)(conv4)

    conv5 = Conv3D(nconv, conv_size, activation='relu', padding='same')(pool4)
    conv5 = Conv3D(nconv, conv_size, activation='relu', padding='same')(conv5)

    up8 = merge([UpSampling3D(size=pool_size)(conv5), conv4], mode='concat', concat_axis=concat_axis)
    conv8 = Conv3D(nconv, conv_size, activation='relu', padding='same')(up8)
    conv8 = Conv3D(nconv, conv_size, activation='relu', padding='same')(conv8)

    up9 = merge([UpSampling3D(size=pool_size)(conv8), conv3], mode='concat', concat_axis=concat_axis)
    conv9 = Conv3D(nconv, conv_size, activation='relu', padding='same')(up9)
    conv9 = Conv3D(nconv, conv_size, activation='relu', padding='same')(conv9)

    up10 = merge([UpSampling3D(size=pool_size)(conv9), conv2], mode='concat', concat_axis=concat_axis)
    conv10 = Conv3D(nconv, conv_size, activation='relu', padding='same')(up10)
    conv10 = Conv3D(nconv, conv_size, activation='relu', padding='same')(conv10)

    up11 = merge([UpSampling3D(size=pool_size)(conv10), conv1], mode='concat', concat_axis=concat_axis)
    conv11 = Conv3D(nconv, conv_size, activation='relu', padding='same')(up11)
    conv11 = Conv3D(nconv, conv_size, activation='relu', padding='same')(conv11)

    # need as many output channel as tissue classes
    conv14 = Conv3D(tissue_classes, (1, 1, 1), activation='sigmoid')(conv11)

    model = Model(input=[inputs], output=[conv14])

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef], sample_weight_mode="temporal")

    return model

def dice_coef(y_true, y_pred):
    """ DICE coefficient: 2TP / (2TP + FP + FN). An additional smoothness term is added to ensure no / 0
    :param y_true: True labels.
    :type: TensorFlow/Theano tensor.
    :param y_pred: Predictions.
    :type: TensorFlow/Theano tensor of the same shape as y_true.
    :return: Scalar DICE coefficient.
    """

    print(np.shape(y_true))
    print(np.shape(y_pred))

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    class_weight = {}
    class_weight[0] = 10  # don't care about background
    class_weight[10] = 70  # CSF
    class_weight[150] = 90  # WM
    class_weight[250] = 100  # GM



    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)  # the 1 is to ensure smoothness

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def to_categorical(y, class_weights=None):
    """Converts a class vector (integers) to binary class matrix.
    Keras function did not support sparse category labels
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    categories = set(np.array(y, dtype="int").ravel())
    num_classes = len(categories)

    cat_shape = np.shape(y)[:-1] + (num_classes,)
    categorical = np.zeros(cat_shape, dtype='b')

    sample_weights = np.zeros(np.shape(y)[:-1], dtype='uint8')

    # print('sample weight shape', np.shape(sample_weights))
    # print("class weights in to_categorical:", class_weights)
    for i, cat in enumerate(categories):
        categorical[..., i] = np.squeeze(np.equal(y, np.ones(np.shape(y))*cat))
        # print('category', cat, 'has', np.sum(categorical[..., i]), 'voxels')
        # test = nib.Nifti1Image(categorical[...,i], np.eye(4))
        # nib.save(test, 'cat' + str(cat) + '.nii.gz')
        if not class_weights == None:
            sample_weights[categorical[..., i] == 1] = class_weights[cat]

    if not class_weights == None:
        return categorical, sample_weights
    else:
        return categorical

def from_categorical(categorical, category_mapping):
    """Combines several binary masks for tissue classes into a single segmentation image
    :param categorical:
    :param category_mapping:
    :return:
    """
    img_shape = np.shape(categorical)[1:-1]
    cat_img = np.argmax(np.squeeze(categorical), axis=3)

    # img = nib.Nifti1Image(cat_img, np.eye(4))
    # nib.save(img, 'cat_img.nii.gz')

    segmentation = np.zeros(img_shape, dtype='uint8')

    for i, cat in enumerate(category_mapping):
        # print('category', cat, 'has', np.sum(categorical[..., i]), 'voxels')
        # print('category', cat, 'has', len(segmentation[cat_img == cat]), 'voxels')
        segmentation[cat_img == 1] = cat

    return segmentation

def batch(indices, class_weights=None):
    """
    :param indices: List of indices into the HDF5 dataset to draw samples from
    :return: (image, label)
    """
    f = h5py.File(input_file)
    images = f['images']
    labels = f['labels']

    while True:
        np.random.shuffle(indices)
        for i in indices:
            # print("class weights in batch", class_weights)
            if not class_weights == None:
                label, sample_weight = to_categorical(labels[i, ...], class_weights=class_weights)
            else:
                label = to_categorical(labels[i, ...])

            yield (images[i, ...][np.newaxis, ...], label[np.newaxis, ...])


if __name__ == "__main__":
    f = h5py.File(input_file)
    images = f['images']
    labels = f['labels']

    training_indices = np.linspace(0, 8)
    validation_indices = [9]
    testing_indices = [10]

    # train_data = np.reshape(images[training_indices], (8, 144, 192, 256, 2))
    # train_labels = np.reshape(labels[training_indices], (8, 144, 192, 256, 1))
    # test_data = np.reshape(images[testing_indices], (1, 144, 192, 256, 2))
    # test_labels = np.reshape(labels[testing_indices], (1, 144, 192, 256, 1))
    # validation_data = np.reshape(images[validation_indices], (1, 144, 192, 256, 2))
    # validation_labels = np.reshape(labels[validation_indices], (1, 144, 192, 256, 1))

    model = segmentation_model()
    model.summary()

    model_checkpoint = ModelCheckpoint(scratch_dir + 'best_seg_model.hdf5', monitor="val_dice_coef", verbose=0,
                                       save_best_only=True, save_weights_only=False, mode='auto')

    # for epoch in range(10):
    #     print("training epoch:", epoch)
    #     for index in range(len(training_indices)):
    #         loss = model.train_on_batch(np.reshape(images[index], (1, 144, 192, 256, 2)), np.reshape(labels[index], (1, 144, 192, 256, 1)))

    # history = model.fit(train_data, train_labels, nb_epoch=10, batch_size=1,
    #                     validation_data=(validation_data, validation_labels), callbacks=[model_checkpoint])




    hist = model.fit_generator(
        batch(training_indices),
        len(training_indices),
        epochs=1,
        verbose=1,
        callbacks=[model_checkpoint],
        validation_data=batch(validation_indices),
        validation_steps=1)

    model.load_weights(scratch_dir + 'best_seg_model.hdf5')

    category_mapping = [0, 10, 150, 250]

    #test image
    predicted = model.predict_generator(batch(testing_indices), steps=1)
    segmentation = from_categorical(predicted, category_mapping)
    test_img = nib.Nifti1Image(segmentation, np.eye(4))
    nib.save(test_img, 'test_image_segmentation.nii.gz')

    #validation image
    predicted = model.predict_generator(batch(validation_indices), steps=1)
    segmentation = from_categorical(predicted, category_mapping)
    val_image = nib.Nifti1Image(segmentation, np.eye(4))
    nib.save(val_image, 'val_image_segmentation.nii.gz')

    epoch_num = range(len(hist.history['dice_coef']))
    dice_train = np.array(hist.history['dice_coef'])
    dice_val = np.array(hist.history['val_dice_coef'])

    plt.clf()
    plt.plot(epoch_num, dice_train, label='DICE Score Training')
    plt.plot(epoch_num, dice_val, label="DICE Score Validation")
    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Score")
    plt.savefig(scratch_dir + 'results.png')
    plt.close()
