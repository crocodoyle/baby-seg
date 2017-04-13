from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, SpatialDropout2D, merge
from keras.layers import Convolution3D, MaxPooling3D, SpatialDropout3D, UpSampling3D

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



def make_iseg():

    training_dir = 'E:/baby-seg/training/'
    testing_dir  = 'E:/baby-seg/testing/'
    scratch_dir = 'E:/'
    numImgs = 23

    x_dim = 144
    y_dim = 192
    z_dim = 256

    f = h5py.File(scratch_dir + 'baby-seg.hdf5', 'w')
    f.create_dataset('images', (numImgs, x_dim, y_dim, z_dim, 2), dtype='float32')
    f.create_dataset('labels', (numImgs, x_dim, y_dim, z_dim, 1), dtype='uint8')

    numSeen = {}
    numSeen['T1'] = 0
    numSeen['T2'] = 0
    numSeen['labels'] = 0

    for filename in os.listdir(training_dir):
        index = int(filename.split('-')[1]) - 1

        if '.img' in filename:
            print(filename)
            if 'T1' in filename:
                f['images'][index, ..., 0] = nib.load(os.path.join(training_dir, filename)).get_data()[:, :, :, 0]
                numSeen['T1'] += 1
            if 'T2' in filename:
                f['images'][index, ..., 1] = nib.load(os.path.join(training_dir, filename)).get_data()[:, :, :, 0]
                numSeen['T2'] += 1
            if 'label' in filename:
                f['labels'][index, ...] = nib.load(os.path.join(training_dir, filename)).get_data()
                numSeen['labels'] += 1

    for filename in os.listdir(testing_dir):
        index = int(filename.split('-')[1]) - 1
        if '.img' in filename:
            print(filename)
            if 'T1' in filename:
                if not '23' in filename:
                    f['images'][index, ..., 0] = nib.load(os.path.join(testing_dir, filename)).get_data()[:, :, :, 0]
                else:
                    f['images'][index, ..., 0] = nib.load(os.path.join(testing_dir, filename)).get_data()[8:-8, :, :, 0]
                numSeen['T1'] += 1
            if 'T2' in filename:
                if not '23' in filename:
                    f['images'][index, ..., 1] = nib.load(os.path.join(testing_dir, filename)).get_data()[:, :, :, 0]
                else:
                    f['images'][index, ..., 1] = nib.load(os.path.join(testing_dir, filename)).get_data()[8:-8, :, :, 0]
            if 'label' in filename:
                f['labels'][index, ...] = nib.load(os.path.join(testing_dir, filename)).get_data()

    f.close()

    return




def batch(indices, labels, n, random_slice=False):
    f = h5py.File(scratch_dir + 'oasis.hdf5', 'r')
    images = f.get('oasis')
    labels = f.get('oasis_labels')

    x_train = np.zeros((n, 1, 180, 217, 180), dtype=np.float32)
    y_train = np.zeros((n, 180, 217, 180), dtype=np.int8)

    while True:
        np.random.shuffle(indices)

        samples_this_batch = 0
        for i, index in enumerate(indices):
            x_train[i%n, :, :, :] = images[index, :-2, :, :-2]
            y_train[i%n, :, :, :] = labels[index, :-2, :, :-2]
            samples_this_batch += 1
            if (i+1) % n == 0:
                yield (x_train, y_train)
                samples_this_batch = 0
            elif i == len(indices)-1:
                yield (x_train[0:samples_this_batch, ...], y_train[0:samples_this_batch, :])
        samples_this_batch = 0

def test_images(model):
    f = h5py.File(scratch_dir + 'oasis.hdf5', 'r')

    images = f.get('oasis')
    labels = f.get('oasis_labels')

    model.predict()


    dice = 0


    return dice

if __name__ == "__main__":

    make_iseg()

    # print "Running segmentation training"
    #
    # train_indices, validation_indices, test_indices, patient_id = load_oasis()

    # model = segmentation_model()
    # model.summary()
    #
    # plot(model, to_file="segmentation_model.png")
    #
    #
    # model_checkpoint = ModelCheckpoint("models/best_segmentation_model.hdf5", monitor="val_acc", verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    #
    # hist = model.fit_generator(batch(train_indices, labels, 2,True), nb_epoch=400, samples_per_epoch=len(train_indices), validation_data=batch(validation_indices, labels, 2), nb_val_samples=len(validation_indices), callbacks=[model_checkpoint], class_weight = {0:.7, 1:.3})
    #
    #
    # model.load_weights('models/best_segmentation_model.hdf5')
