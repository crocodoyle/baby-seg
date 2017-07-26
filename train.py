from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, \
    SpatialDropout2D, merge, Reshape, Cropping3D
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, UpSampling3D, ZeroPadding3D
from keras.layers import concatenate, add, multiply
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

# from keras.utils.visualize_util import plot
from keras.callbacks import Callback
from keras import backend as K

from keras.utils import to_categorical as to_cat

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

from numpy.lib.stride_tricks import as_strided
from skimage.util import view_as_windows

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

category_mapping = [0, 10, 150, 250]
img_shape = (144, 192, 128)

n_tissues = 4

patch_shape = (80, 80, 80)

class SegVisCallback(Callback):

    def on_train_begin(self, logs={}):
        self.segmentations = []

        self.f = h5py.File(input_file)
        self.images = f['images']

    def on_epoch_end(self, batch, logs={}):
        model = self.model

        predicted = model.predict(self.images[9, ...][np.newaxis, ...], batch_size=1)
        segmentation = from_categorical(predicted, category_mapping)

        slice = segmentation[:, :, 64].T
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

        f = h5py.File(input_file)
        self.images = f['images']
        self.labels = f['labels']

    def on_epoch_end(self, batch, logs={}):
        model = self.model

        conf = np.zeros((len(category_mapping),len(category_mapping)))

        print('\n')
        for i in range(1):
            predicted = model.predict(self.images[i,...][np.newaxis, ...], batch_size=1)
            segmentation = from_categorical(predicted, category_mapping)

            y_true = self.labels[i, ..., 0].flatten()
            y_pred = segmentation.flatten()

            conf = confusion_matrix(y_true, y_pred)

            # print(conf)

        print("------")
        print('confusion matrix:', category_mapping)
        print(conf)
        print("------")

        self.confusion.append(conf)

    def on_train_end(self, logs={}):
        tissue_classes = ["BG", "CSF", "GM", "WM"]

        for epoch, conf in enumerate(self.confusion):
            filename = os.path.join(scratch_dir, 'confusion', 'confusion_' + str(epoch).zfill(4) + '.png')
            save_confusion_matrix(conf, tissue_classes, filename)

        images = []
        for filename in os.listdir(os.path.join(scratch_dir, 'confusion')):
            if '.png' in filename and not 'results' in filename:
                images.append(plt.imread(os.path.join(scratch_dir, 'confusion', filename)))

            # imageio.mimsave(os.path.join(scratch_dir, 'confusion', 'confusion.gif'), images)


def model_checkpoint(filename):
    return ModelCheckpoint(scratch_dir + 'filename', save_best_only=True, save_weights_only=True)


def lr_scheduler(model):
    # reduce learning rate by factor of 10 every 100 epochs
    def schedule(epoch):
        new_lr = K.get_value(model.optimizer.lr)

        if epoch % 200 == 0:
            new_lr = new_lr / 2

        return new_lr

    scheduler = LearningRateScheduler(schedule)
    return scheduler

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

def convnet():
    inputs = Input(shape=(patch_shape + (2,)))

    conv_size = (3, 3, 3)

    conv1 = Conv3D(64, conv_size, activation='relu', strides=(2, 2, 2), padding='valid')(inputs)
    conv1 = Conv3D(64, conv_size, activation='relu', padding='valid')(conv1)
    drop1 = Dropout(0.1)(conv1)
    norm1 = BatchNormalization()(drop1)
    conv2 = Conv3D(64, conv_size, activation='relu', padding='valid')(norm1)
    drop2 = Dropout(0.2)(conv2)
    norm2 = BatchNormalization()(drop2)
    conv3 = Conv3D(64, conv_size, activation='relu', padding='valid')(norm2)
    drop3 = Dropout(0.3)(conv3)
    norm3 = BatchNormalization()(drop3)
    conv4 = Conv3D(64, conv_size, activation='relu', padding='valid')(norm3)
    drop4 = Dropout(0.4)(conv4)
    norm4 = BatchNormalization()(drop4)
    conv5 = Conv3D(64, conv_size, activation='relu', padding='valid')(norm4)
    drop5 = Dropout(0.5)(conv5)
    norm5 = BatchNormalization()(drop5)
    conv6 = Conv3D(64, conv_size, activation='relu', padding='valid')(norm5)
    drop6 = Dropout(0.5)(conv6)
    norm6 = BatchNormalization()(drop6)

    flat = Flatten()(norm6)

    fc1 = Dense(10)(flat)
    drop_fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(10)(drop_fc1)
    drop_fc2 = Dropout(0.5)(fc2)

    outputs = Dense(n_tissues, activation='softmax')(drop_fc2)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

def unet_patch():
    """
    3D U-net model, using very small convolutional kernels
    """

    big_conv_size = (5, 5, 5)
    small_conv_size = (3, 3, 3)
    mini_conv_size = (1, 1, 1)

    pool_size = (2, 2, 2)

    inputs = Input(shape=(80, 80, 80, 2))

    conv1 = Conv3D(4, big_conv_size, activation='relu', padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(bn1)

    conv2 = Conv3D(16, big_conv_size, activation='relu', padding='same')(pool1)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(bn2)

    conv3 = Conv3D(16, big_conv_size, activation='relu', padding='same')(pool2)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(bn3)

    conv4 = Conv3D(16, big_conv_size, activation='relu', padding='same')(pool3)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=pool_size)(bn4)

    # conv5 = Conv3D(32, big_conv_size, activation='relu', padding='same')(pool4)
    # bn5 = BatchNormalization()(conv5)
    # pool5 = MaxPooling3D(pool_size=pool_size)(bn5)

    conv6 = Conv3D(32, big_conv_size, activation='relu', padding='same')(pool4)
    conv7 = Conv3D(32, small_conv_size, activation='relu', padding='same')(pool4)
    conv8 = Conv3D(32, mini_conv_size, activation='relu', padding='same')(pool4)
    nadir = add([conv6, conv7, conv8])
    bn8 = BatchNormalization()(nadir)

    # skip9 = concatenate([pool5, bn8])
    # up9 = UpSampling3D(size=pool_size)(skip9)
    # conv9 = Conv3D(64, big_conv_size, activation='relu', padding='same')(up9)
    # bn9 = BatchNormalization()(conv9)

    skip10 = concatenate([pool4, bn8])
    up10 = UpSampling3D(size=pool_size)(skip10)
    conv10 = Conv3D(64, big_conv_size, activation='relu', padding='same')(up10)
    bn10 = BatchNormalization()(conv10)

    skip11 = concatenate([pool3, bn10])
    up11 = UpSampling3D(size=pool_size)(skip11)
    conv11 = Conv3D(64, big_conv_size, activation='relu', padding='same')(up11)
    bn11 = BatchNormalization()(conv11)

    skip12 = concatenate([pool2, bn11])
    up12 = UpSampling3D(size=pool_size)(skip12)
    conv12 = Conv3D(128, big_conv_size, activation='relu', padding='same')(up12)
    bn12 = BatchNormalization()(conv12)

    skip13 = concatenate([pool1, bn12])
    up13 = UpSampling3D(size=pool_size)(skip13)
    conv13 = Conv3D(256, big_conv_size, activation='relu', padding='same')(up13)
    bn13 = BatchNormalization()(conv13)

    conv14 = Conv3D(256, big_conv_size, activation='relu', padding='same')(bn13)
    bn14 = BatchNormalization()(conv14)
    conv15 = Conv3D(128, small_conv_size, activation='relu', padding='same')(bn14)
    bn15 = BatchNormalization()(conv15)
    conv16 = Conv3D(128, mini_conv_size, activation='relu', padding='same')(bn15)
    drop16 = Dropout(0.5)(conv16)
    bn17 = BatchNormalization()(drop16)

    # need as many output channel as tissue classes
    conv17 = Conv3D(n_tissues, (1, 1, 1), activation='softmax', padding='valid')(bn17)

    crop = Cropping3D(((8, 8), (8, 8), (8, 8)))(conv17)

    model = Model(input=[inputs], output=[crop])

    return model


def fractal_block(nb_filter, b, c, drop_path, dropout=0):
    from fractalnet import fractal_net

    conv_size = (3, 3, 3)

    def f(input):
        return fractal_net(b=b, c=c, conv=b*[(nb_filter, conv_size)], drop_path=drop_path, dropout=b*[dropout])(input)
    return f


def train_tl():
    from neuroembedding import autoencoder, encoder, t_net, tl_net

    training_indices = list(range(0, 10))
    validation_indices = [9]
    testing_indices = list(range(10, 24))
    ibis_indices = list(range(24, 53))

    autoencoder = autoencoder()

    sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)

    autoencoder.compile(optimizer=sgd, loss='categorical_crossentropy')

    hist_one = autoencoder.fit_generator(
        label_batch(training_indices),
        len(training_indices),
        epochs=200,
        verbose=1,
        callbacks=[lr_scheduler(autoencoder), model_checkpoint('autoencoder.hdf5')],
        validation_data=[label_batch(validation_indices)],
        validation_steps=[len(validation_indices)]
    )       #training autoencoder should be complete here

    enc_model = encoder()
    enc_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    enc_model.load_weights(scratch_dir + 'autoencoder.hdf5', by_name=True)

    encoded_shape = enc_model.get_output_shape_at('enc')
    print('shape of encoded label space', encoded_shape)

    f2 = h5py.File(scratch_dir + 'encoded_labels.hdf5', 'w')
    f2.create_dataset('label_encoding', (len(training_indices) + len(validation_indices) + len(ibis_indices), encoded_shape), dtype='float32')
    label_encoding = f2['label_encoding']

    f = h5py.File(input_file)
    labels = f['labels']
    images = f['images']

    for i in training_indices + validation_indices + ibis_indices:
        predicted = enc_model.predict(labels[i, ...][np.newaxis, ...], batch_size=1)
        segmentation = from_categorical(predicted, category_mapping)
        label_encoding[i, ...] = segmentation

    t_net = t_net()
    t_net.load_weights(scratch_dir + 'autoencoder.hdf5', by_name=True)
    t_net.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    hist_two = t_net.fit_generator(
        batch(training_indices),
        len(training_indices),
        epochs=500,
        verbose=1,
        callbacks=[lr_scheduler, model_checkpoint('t_net.hdf5')],
        validation_data=batch(validation_indices),
        validation_steps=len(validation_indices)
    )

    tl_net = tl_net()
    tl_net.load_weights(scratch_dir + 't_net.hdf5')
    tl_net.compile(optimizer=Adam(lr=1e-8), loss=dice_coef_loss, metrics=[dice_coef])

    hist_three = tl_net.fit(
        [images[training_indices, :, :, :][np.newaxis, ...], labels[training_indices, :, :, :][np.newaxis, ...]],
        [labels[training_indices, :, :, :][np.newaxis, ...]],
        batch_size=1,
        epochs=100,
        verbose=1,
        callbacks=[lr_scheduler, model_checkpoint('tl_net.hdf5')],
        validation_data=[images[validation_indices, :, :, :][np.newaxis, ...], labels[validation_indices, :, :, :][np.newaxis, ...]]
    )


###########################
#
#   LOSS FUNCTIONS
#
###########################


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

    # category_weight = [1.35, 17.85, 8.27*10, 11.98*10]

    category_weight = [1, 1, 1, 1]

    for i, (c, w) in enumerate(zip(category_mapping, category_weight)):
        score += w*(2.0 * K.sum(y_true[..., i] * y_pred[..., i]) / (K.sum(y_true[..., i]) + K.sum(y_pred[..., i])))

    return score / np.sum(category_weight)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def final_dice_score(y_true, y_pred):
    dice = {}
    for i, c in enumerate(zip(category_mapping)):
        dice[str(c)] = (2.0 * K.sum(y_true[..., i] * y_pred[..., i]) / (K.sum(y_true[..., i]) + K.sum(y_pred[..., i])))

    return dice


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

def patch_label_categorical(y):
    n_samples = y.shape[0]
    cat_shape = (n_samples,) + (n_tissues,)
    categorical = np.zeros(cat_shape, dtype='b')

    for i, cat in enumerate(category_mapping):
        categorical[..., i] = np.equal(y[..., 0], np.ones(np.shape(y[..., 0]))*cat)

    return categorical

def from_categorical(categorical, category_mapping):
    """Combines several binary masks for tissue classes into a single segmentation image
    :param categorical:
    :param category_mapping:
    :return:
    """
    img_shape = np.shape(categorical)[:-1]
    cat_img = np.argmax(categorical, axis=-1)

    segmentation = np.zeros(img_shape, dtype='uint8')

    for i, cat in enumerate(category_mapping):
        segmentation[cat_img == i] = cat

    return segmentation

def unet_patch_gen(indices, n, test_mode=False):
    f = h5py.File(input_file)
    images = f['images']
    labels = f['labels']

    while True:
        np.random.shuffle(indices)
        for i in indices:
            t1_image = np.pad(np.asarray(images[i, ..., 0], dtype='float32'), ((8, 8), (8, 8), (8, 8)), 'constant')
            t2_image = np.pad(np.asarray(images[i, ..., 1], dtype='float32'), ((8, 8), (8, 8), (8, 8)), 'constant')

            true_labels = labels[i, ..., 0]

            # print(true_labels.shape)

            # patches_x = np.zeros((t1_strided.shape[0]*t1_strided.shape[1]*t1_strided.shape[2],) + t1_image.shape + (2,), dtype='float32')
            # patches_y = np.zeros((n_tissues) + patch_shape + )

            patches_x = np.zeros(((n,) + patch_shape + (2,)), dtype='float32')
            patches_y_ints = np.zeros((n,) + (64, 64, 64) + (1,), dtype='uint8')

            for j in range(n):
                x = np.random.randint(0, img_shape[0] - 80) + 8
                y = np.random.randint(0, img_shape[1] - 80) + 8
                z = np.random.randint(0, img_shape[2] - 80) + 8

                patches_x[j, ..., 0] = t1_image[x:x+80, y:y+80, z:z+80]
                patches_x[j, ..., 1] = t2_image[x:x+80, y:y+80, z:z+80]

                if not test_mode:
                    patches_y_ints[j, ..., 0] = true_labels[x-8:x-8+64, y-8:y-8+64, z-8:z-8+64]

            if test_mode:
                yield (patches_x)

            else:
                patches_y = to_categorical(patches_y_ints)
                yield (patches_x, patches_y)


def batch(indices, augmentMode=None):
    """
    :param indices: List of indices into the HDF5 dataset to draw samples from
    :return: (image, label)
    """
    f = h5py.File(input_file)
    images = f['images']
    labels = f['labels']

    return_imgs = np.zeros(img_shape + (2,))

    while True:
        np.random.shuffle(indices)
        for i in indices:
            t1_image = np.asarray(images[i, ..., 0], dtype='float32')
            t2_image = np.asarray(images[i, ..., 1], dtype='float32')

            try:
                true_labels = labels[i, ..., 0]

                if augmentMode is not None:
                    if 'flip' in augmentMode:
                        # flip images
                        if np.random.rand() > 0.5:
                            t1_image = np.flip(t1_image, axis=0)
                            t2_image = np.flip(t2_image, axis=0)
                            true_labels = np.flip(true_labels, axis=0)

                    if 'affine' in augmentMode:

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
                        # ratio_img = affine_transform(ratio_img, trans_mat, cval=10)

                        true_labels = affine_transform(true_labels, trans_mat, order=0, cval=10) # nearest neighbour for labels

                return_imgs[..., 0] = t1_image
                return_imgs[..., 1] = t2_image

                label = to_categorical(np.reshape(true_labels, true_labels.shape + (1,)))

                yield (return_imgs[np.newaxis, ...], label[np.newaxis, ...])

            except ValueError:
                print('some sort of value error occurred')
                # print(images[i, :, :, 80:-48][np.newaxis, ...].shape)
                yield (return_imgs[np.newaxis, ...])


def label_batch(indices):
    f = h5py.File(input_file)
    labels = f['labels']

    while True:
        np.shuffle(indices)

        for i in indices:
            true_labels = labels[i, ..., 0]
            label = to_categorical(np.reshape(true_labels, true_labels.shape + (1,)))

            yield (label[np.newaxis, ...], label[np.newaxis, ...])


def predict_whole_image(index):
    model = unet_patch()
    model.load_weights(scratch_dir + 'unet-3d-patch-iseg2017.hdf5')

    prediction = np.zeros((192, 192, 192, 4), dtype='uint8')
    orig = np.zeros((192, 192, 192), dtype='float32')

    f = h5py.File(input_file)
    images = f['images']

    test_image = images[index, ...]
    print('test img shape:', test_image.shape)

    test_image = np.pad(test_image, ((8, 56), (8, 8), (8, 8), (0, 0)), mode='constant')
    print('test img shape:', test_image.shape)

    # print('images to predict:', input_images.shape)

    for i in range(test_image.shape[0] // 64):
        for j in range(test_image.shape[1] // 64):
            for k in range(test_image.shape[2] // 64):
                try:
                    input_image = test_image[(i*64):(i*64)+80, (j*64):(j*64)+80, (k*64):(k*64)+80][np.newaxis, ...]

                    orig[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64] = test_image[(i*64):(i*64)+80, (j*64):(j*64)+80, (k*64):(k*64)+80][8:-8, 8:-8, 8:-8, 0]

                    print('x range', i*64, i*64+80, 'y range', j*64, j*64+80, 'z range', k*64, k*64+80)
                    print('x dest', i*64, (i+1)*64, 'y dest', j*64, (j+1)*64, 'z dest', k*64, (k+1)*64)

                    prediction[j*64:(j+1)*64, i*64:(i+1)*64, k*64:(k+1)*64] = model.predict(input_image)
                except IndexError as e:
                    print('bad index', e)

    img = nib.Nifti1Image(orig, np.eye(4))
    nib.save(img, scratch_dir + 'test.nii.gz')

    segmentation = from_categorical(np.pad(prediction[:-48, :, :], ((0, 0), (0, 0), (80, 48), (0, 0)), mode='constant'), category_mapping)

    # int_predictions = np.argmax(np.pad(prediction[:-48, :, :], ((0, 0), (0, 0), (80, 48), (0, 0)), mode='constant'), axis=-1)
    # category_predictions = [category_mapping[i] for i in int_predictions]

    # segmentation = np.asarray(np.reshape(category_predictions, img_shape), dtype='uint8')

    return segmentation

def predict_images_with_patches(validation_indices, testing_indices):
    affine = np.eye(4)
    affine[0, 0] = -1
    affine[1, 1] = -1

    for i in validation_indices + testing_indices:
        predicted = predict_whole_image(i)

        segmentation_padded = np.pad(predicted, pad_width=((0, 0), (0, 0), (80, 48)), mode='constant', constant_values=0)
        print(predicted.shape)
        image = nib.Nifti1Image(segmentation_padded[..., np.newaxis], affine)
        nib.save(image, scratch_dir + 'babylabels' + str(i+1).zfill(2) + '.nii.gz')

        if i in validation_indices:
            # print(final_dice_score(labels[i, ..., 0], segmentation))
            print(confusion_matrix(labels[i, ..., 0].flatten(), predicted.flatten()))

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


def train_unet():
    f = h5py.File(input_file)
    images = f['images']
    labels = f['labels']

    training_indices = list(range(9))
    validation_indices = [9]
    testing_indices = list(range(10, 23))
    ibis_indices = list(range(24, 72))

    # training_indices = training_indices + ibis_indices

    print('training images:', training_indices)
    print('validation images:', validation_indices)
    print('testing images:', testing_indices)
    print('ibis images:', ibis_indices)

    affine = np.eye(4)
    affine[0, 0] = -1
    affine[1, 1] = -1

    # model = segmentation_model()
    model = unet_patch()

    sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
    adam = Adam()

    model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])

    model.summary()

    # print('model', dir(model))

    model_checkpoint = ModelCheckpoint(scratch_dir + 'best_patch_unet_model.hdf5', monitor="val_dice_coef",
                                       save_best_only=True, save_weights_only=False, mode='max')
    confusion_callback = ConfusionCallback()
    segvis_callback = SegVisCallback()
    tensorboard = TensorBoard(scratch_dir)

    # train without augmentation (easier)
    hist = model.fit_generator(
        unet_patch_gen(training_indices, 1),
        len(training_indices),
        epochs=1,
        verbose=1,
        callbacks=[model_checkpoint, tensorboard],
        validation_data=unet_patch_gen(validation_indices, 1),
        validation_steps=len(validation_indices))

    model.load_weights(scratch_dir + 'best_patch_unet_model.hdf5')
    model.save(scratch_dir + 'unet-3d-patch-iseg2017.hdf5')

    for i in training_indices + validation_indices + testing_indices:
        predicted = predict_whole_image(i)
        image = nib.Nifti1Image(predicted, affine)
        nib.save(image, scratch_dir + 'babylabels' + str(i+1).zfill(2) + '.nii.gz')


    visualize_training_dice(hist)


if __name__ == "__main__":
    f = h5py.File(input_file)
    images = f['images']
    labels = f['labels']

    train_unet()