import numpy as np
import h5py

import os
import nibabel as nib

import pickle as pkl

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import argparse

def make_iseg():
    """
    Reads data from the MICCAI 2017 Grand Challenge (iSeg 2017) and creates an HDF5 dataset.
    Subject 23 has a different size than the others.
    """
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

if __name__ == "__main__":
    make_iseg()
