from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, \
    SpatialDropout2D, merge, Reshape
from keras.layers import Conv3D, MaxPooling3D, SpatialDropout3D, UpSampling3D, ZeroPadding3D
from keras.layers.merge import Concatenate
from keras.layers import concatenate
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from keras.utils.visualize_util import plot
from keras.callbacks import Callback
from keras import backend as K

import tensorflow as tf

from sklearn.metrics import confusion_matrix

# configures TensorFlow to not try to grab all the GPU memory
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

import numpy as np
import h5py
import itertools
import imageio


import os
import nibabel as nib

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import argparse

from scipy.ndimage.interpolation import affine_transform

import transformations as t
import math

scratch_dir = '/data1/data/iSeg-2017/'
input_file = scratch_dir + 'baby-seg.hdf5'

category_mapping = [10, 150, 250]
img_shape = (144, 192, 256)

class SegVisCallback(Callback):

    def on_train_begin(self, logs={}):
        self.segmentations = []

        self.f = h5py.File(input_file)
        self.images = f['images']

    def on_epoch_end(self, batch, logs={}):
        model = self.model

        predicted = model.predict(self.images[0, ...][np.newaxis, ...], batch_size=1)
        segmentation = from_categorical(predicted, category_mapping)

        slice = segmentation[:, :, 128].T
        self.segmentations.append(slice)

    def on_train_end(self, logs={}):

        for i, seg in enumerate(self.segmentations):
            plt.imsave(os.path.join(scratch_dir, 'segmentations', 'example_segmentation_' + str(i).zfill(4) + '.png'), seg)

        images = []
        for filename in sorted(os.listdir(os.path.join(scratch_dir, 'segmentations'))):
            if '.png' in filename:
                images.append(plt.imread(os.path.join(scratch_dir, 'segmentations', filename)))

        imageio.mimsave(os.path.join(scratch_dir, 'segmentations', 'segmentation.gif'), images)


class ConfusionCallback(Callback):

    def on_train_begin(self, logs={}):
        self.confusion = []

        self.f = h5py.File(input_file)
        self.images = f['images']
        self.labels = f['labels']

    def on_epoch_end(self, batch, logs={}):
        model = self.model

        conf = np.zeros((len(category_mapping),len(category_mapping)))

        print('\n')
        for i in range(1):
            predicted = model.predict(self.images[i,...][np.newaxis, ...], batch_size=1)
            segmentation = from_categorical(predicted, category_mapping)

            y_true = self.labels[i,...,0].flatten()
            y_pred = segmentation.flatten()

            conf = confusion_matrix(y_true, y_pred)

            # print(conf)

        print("------")
        print('confusion matrix:', category_mapping)
        print(conf)
        print("------")

        self.confusion.append(conf)

    def on_train_end(self, logs={}):
        tissue_classes = ["CSF", "GM", "WM"]

        for epoch, conf in enumerate(self.confusion):
            filename = os.path.join(scratch_dir, 'confusion', 'confusion_' + str(epoch).zfill(4) + '.png')
            save_confusion_matrix(conf, tissue_classes, filename)

        images = []
        for filename in os.listdir(os.path.join(scratch_dir, 'confusion')):
            if '.png' in filename and not 'results' in filename:
                images.append(plt.imread(os.path.join(scratch_dir, 'confusion', filename)))

            imageio.mimsave(os.path.join(scratch_dir, 'confusion', 'confusion.gif'), images)


def save_confusion_matrix(cm, classes, filename,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """This function plots the confusion matrix."""
    plt.clf()

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename, bbox_inches='tight')

def segmentation_model():
    """
    3D U-net model, using very small convolutional kernels
    """
    tissue_classes = 3

    conv_size = (3, 3, 3)
    pool_size = (2, 2, 2)

    inputs = Input(shape=(144, 192, 256, 3))

    conv1 = Conv3D(16, conv_size, activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16, conv_size, activation='relu', padding='same')(conv1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(bn1)

    conv2 = Conv3D(32, conv_size, activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, conv_size, activation='relu', padding='same')(conv2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(bn2)

    conv3 = Conv3D(32, conv_size, activation='relu', padding='same')(pool2)
    conv3 = Conv3D(32, conv_size, activation='relu', padding='same')(conv3)
    drop3 = Dropout(0.3)(conv3)
    bn3 = BatchNormalization()(drop3)
    pool3 = MaxPooling3D(pool_size=pool_size)(bn3)

    conv4 = Conv3D(64, conv_size, activation='relu', padding='same')(pool3)
    drop4 = Dropout(0.4)(conv4)
    conv4 = Conv3D(64, conv_size, activation='relu', padding='same')(drop4)
    drop4 = Dropout(0.4)(conv4)
    bn4 = BatchNormalization()(drop4)
    pool4 = MaxPooling3D(pool_size=pool_size)(bn4)

    conv5 = Conv3D(64, conv_size, activation='relu', padding='same')(pool4)
    drop5 = Dropout(0.5)(conv5)
    conv5 = Conv3D(64, conv_size, activation='relu', padding='same')(drop5)
    drop5 = Dropout(0.5)(conv5)
    bn5 = BatchNormalization()(drop5)
    pool5 = MaxPooling3D(pool_size=pool_size)(bn5)

    conv6 = Conv3D(64, conv_size, activation='relu', padding='same')(pool5)
    drop6 = Dropout(0.5)(conv6)
    conv6 = Conv3D(64, conv_size, activation='relu', padding='same')(drop6)
    drop6 = Dropout(0.5)(conv6)
    bn6 = BatchNormalization()(drop6)
    # pool6 = MaxPooling3D(pool_size=pool_size)(bn6)

    up7 = UpSampling3D(size=pool_size)(bn6)
    zp7 = ZeroPadding3D(padding=((1, 1), (0, 0), (0, 0)))(up7)
    concat7 = concatenate([zp7, bn5])
    conv7 = Conv3D(32, conv_size, activation='relu', padding='same')(concat7)
    drop7 = Dropout(0.4)(conv7)
    conv7 = Conv3D(32, conv_size, activation='relu', padding='same')(drop7)
    drop7 = Dropout(0.4)(conv7)
    bn7 = BatchNormalization()(drop7)

    up8 = UpSampling3D(size=pool_size)(bn7)
    concat8 = concatenate([up8, bn4])
    conv8 = Conv3D(32, conv_size, activation='relu', padding='same')(concat8)
    drop8 = Dropout(0.4)(conv8)
    conv8 = Conv3D(32, conv_size, activation='relu', padding='same')(drop8)
    drop8 = Dropout(0.4)(conv8)
    bn8 = BatchNormalization()(drop8)

    up9 = UpSampling3D(size=pool_size)(bn8)
    concat9 = concatenate([up9, bn3])
    conv9 = Conv3D(32, conv_size, activation='relu', padding='same')(concat9)
    conv9 = Conv3D(32, conv_size, activation='relu', padding='same')(conv9)
    drop8 = Dropout(0.3)(conv9)
    bn9 = BatchNormalization()(drop8)

    up10 = UpSampling3D(size=pool_size)(bn9)
    concat10 = concatenate([up10, bn2])
    conv10 = Conv3D(32, conv_size, activation='relu', padding='same')(concat10)
    conv10 = Conv3D(32, conv_size, activation='relu', padding='same')(conv10)
    bn10 = BatchNormalization()(conv10)

    up11 = UpSampling3D(size=pool_size)(bn10)
    concat11 = concatenate([up11, bn1])
    conv11 = Conv3D(16, conv_size, activation='relu', padding='same')(concat11)
    conv11 = Conv3D(16, conv_size, activation='relu', padding='same')(conv11)
    bn11 = BatchNormalization()(conv11)

    # need as many output channel as tissue classes
    conv14 = Conv3D(tissue_classes, (1, 1, 1), activation='softmax', padding='valid')(bn11)

    model = Model(input=[inputs], output=[conv14])

    model.compile(optimizer=Adam(lr=1e-4, decay=1e-7), loss=dice_coef_loss, metrics=[dice_coef])
    # sgd = SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])

    return model

def dice_coef(y_true, y_pred):
    """ DICE coefficient: 2TP / (2TP + FP + FN). An additional smoothness term is added to ensure no / 0
    :param y_true: True labels.
    :type: TensorFlow/Theano tensor.
    :param y_pred: Predictions.
    :type: TensorFlow/Theano tensor of the same shape as y_true.
    :return: Scalar DICE coefficient.
    """
    #exclude the background class from DICE calculation

    score = 0

    category_weight = [0.00001, 1.0, 0.9]

    for i, (c, w) in enumerate(zip(category_mapping, category_weight)):
        score += w*(2.0 * K.sum(y_true[..., i] * y_pred[..., i]) / (K.sum(y_true[..., i]) + K.sum(y_pred[..., i])))

    return score / np.sum(category_weight)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def to_categorical(y):
    """Converts a class vector (integers) to binary class matrix.
    Keras function did not support sparse category labels
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    categories = sorted(set(np.array(y, dtype="uint8").flatten()))
    num_classes = len(categories)
    # print(categories)

    cat_shape = np.shape(y)[:-1] + (num_classes,)
    categorical = np.zeros(cat_shape, dtype='b')

    for i, cat in enumerate(categories):
        categorical[..., i] = np.equal(y[..., 0], np.ones(np.shape(y[..., 0]))*cat)
        # categorical[y == cat] = 1

    return categorical

def from_categorical(categorical, category_mapping):
    """Combines several binary masks for tissue classes into a single segmentation image
    :param categorical:
    :param category_mapping:
    :return:
    """
    img_shape = np.shape(categorical)[1:-1]
    cat_img = np.argmax(np.squeeze(categorical), axis=-1)

    segmentation = np.zeros(img_shape, dtype='uint8')

    for i, cat in enumerate(category_mapping):
        segmentation[cat_img == i] = cat

    return segmentation

def batch(indices, augment=False):
    """
    :param indices: List of indices into the HDF5 dataset to draw samples from
    :return: (image, label)
    """
    f = h5py.File(input_file)
    images = f['images']
    labels = f['labels']

    return_imgs = np.zeros(images.shape[1:-1] + (3,))

    while True:
        np.random.shuffle(indices)
        for i in indices:
            try:
                t1_image = np.asarray(images[i, ..., 0], dtype='float32')
                t2_image = np.asarray(images[i, ..., 1], dtype='float32')
                ratio_img = np.asarray(images[i, ..., 2], dtype='float32')

                true_labels = labels[i, ..., 0]

                if augment:
                    # flip images
                    # if np.random.rand() > 0.5:
                    #     mid = (72, 96, 128)
                    #     normal = (0, 1, 0)
                    #
                    #     reflect_mat = t.reflection_matrix(mid, normal)
                    #     reflect_mat = reflect_mat[0:-1, 0:-1]
                    #     # print('reflection matrix:', reflect_mat)
                    #     # print(reflect_mat.shape)
                    #
                    #     t1_image = affine_transform(t1_image, reflect_mat)
                    #     t2_image = affine_transform(t2_image, reflect_mat)
                    #     true_labels = affine_transform(true_labels, reflect_mat, order=0) # nearest neighbour for labels

                    if np.random.rand() > 0.5:
                        scale = 1 + (np.random.rand(3) - 0.5) * 0.1 # up to 5% scale
                    else:
                        scale = None

                    if np.random.rand() > 0.5:
                        shear = (np.random.rand(3) - 0.5) * 0.2 # sheer of up to 10%
                    else:
                        shear = None

                    if np.random.rand() > 0.5:
                        angles = (np.random.rand(3) - 0.5) * 0.1 * 2*math.pi # rotation up to 5 degrees
                    else:
                        angles = None

                    trans_mat = t.compose_matrix(scale=scale, shear=shear, angles=angles)
                    trans_mat = trans_mat[0:-1, 0:-1]

                    t1_image = affine_transform(t1_image, trans_mat, cval=10)
                    t2_image = affine_transform(t2_image, trans_mat, cval=10)
                    ratio_img = affine_transform(ratio_img, trans_mat, cval=10)

                    true_labels = affine_transform(true_labels, trans_mat, order=0, cval=10) # nearest neighbour for labels

                return_imgs[..., 0] = t1_image
                return_imgs[..., 1] = t2_image
                return_imgs[..., 2] = ratio_img

                label = to_categorical(np.reshape(true_labels, true_labels.shape + (1,)))
                # print(label.shape)

                # print(return_imgs[np.newaxis,...].shape, label[np.newaxis, ...].shape)
                yield (return_imgs[np.newaxis, ...], label[np.newaxis, ...])

            except ValueError:
                print('some sort of value error occurred')
                print(images[i, ...][np.newaxis, ...].shape)
                yield (images[i, ...][np.newaxis, ...])

def visualize_training_dice(hist):
    epoch_num = range(len(hist.history['dice_coef']))
    dice_train = np.array(hist.history['dice_coef'])
    dice_val = np.array(hist.history['val_dice_coef'])

    plt.clf()
    plt.plot(epoch_num, dice_train, label='DICE Score - Training')
    plt.plot(epoch_num, dice_val, label="DICE Score - Validation")
    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(scratch_dir + 'results.png')
    plt.close()

if __name__ == "__main__":
    f = h5py.File(input_file)
    images = f['images']
    labels = f['labels']

    output_shape = (144, 192, 256, 3)

    training_indices = list(range(9))
    validation_indices = [9]
    testing_indices = list(range(10, 23))

    print('training images:', training_indices)
    print('validation images:', validation_indices)
    print('testing images:', testing_indices)

    affine = np.eye(4)
    affine[0, 0] = -1
    affine[1, 1] = -1

    model = segmentation_model()
    model.summary()

    model_checkpoint = ModelCheckpoint(scratch_dir + 'best_seg_model.hdf5', monitor="val_dice_coef", verbose=1,
                                       save_best_only=True, save_weights_only=False, mode='max')

    confusion_callback = ConfusionCallback()
    segvis_callback = SegVisCallback()

    # train without augmentation (easier)
    hist = model.fit_generator(
        batch(training_indices),
        len(training_indices),
        epochs=100,
        verbose=1,
        callbacks=[model_checkpoint, confusion_callback, segvis_callback],
        validation_data=batch(validation_indices),
        validation_steps=len(validation_indices))

    # train the rest of the way with data augmentation
    hist = model.fit_generator(
        batch(training_indices, augment=True),
        len(training_indices),
        epochs=600,
        verbose=1,
        callbacks=[model_checkpoint, confusion_callback, segvis_callback],
        validation_data=batch(validation_indices),
        validation_steps=len(validation_indices))

    model.load_weights(scratch_dir + 'best_seg_model.hdf5')
    model.save(scratch_dir + 'unet-3d-iseg2017.hdf5')

    for i in training_indices + validation_indices + testing_indices:
        predicted = model.predict(images[i,...][np.newaxis, ...], batch_size=1)
        segmentation = from_categorical(predicted, category_mapping)
        image = nib.Nifti1Image(segmentation, affine)
        nib.save(image, scratch_dir + 'babylabels' + str(i).zfill(2) + '.nii.gz')

        print(labels[i,..., 0].shape, segmentation.shape)
        print('confusion matrix for', str(i))
        print(confusion_matrix(labels[i, ..., 0].flatten(), segmentation.flatten()))

    visualize_training_dice(hist)