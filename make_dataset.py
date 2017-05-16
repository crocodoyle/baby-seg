import numpy as np
import h5py

import os
import nibabel as nib

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
    f.create_dataset('images', (numImgs, x_dim, y_dim, z_dim, 3), dtype='float16')
    f.create_dataset('labels', (numImgs, x_dim, y_dim, z_dim, 1), dtype='uint8')

    numSeen = {}
    numSeen['T1'] = 0
    numSeen['T2'] = 0
    numSeen['labels'] = 0

    maxVal = {}
    maxVal['T1'] = 0
    maxVal['T2'] = 0

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

    for i in range(23):
        t1 = f['images'][i, ..., 0]
        t2 = f['images'][i, ..., 1]

        max_t1 = np.max(t1)
        max_t2 = np.max(t2)

        if max_t1 > maxVal['T1']:
            maxVal['T1'] = max_t1
        if max_t2 > maxVal['T2']:
            maxVal['T2'] = max_t2

    print(maxVal)
    for i in range(23):
        f['images'][i, ..., 0] = np.divide(f['images'][i, ..., 0], maxVal['T1'])
        f['images'][i, ..., 1] = np.divide(f['images'][i, ..., 1], maxVal['T2'])
        f['images'][i, ..., 2] = np.divide(f['images'][i, ..., 0], f['images'][i, ..., 1])

    # pretend background is CSF for easier training
    for i in range(23):
        label_img = f['labels'][i, ..., 0]
        label_img[label_img == 0] = 10
        f['labels'][i, ..., 0] = label_img

    f.close()

    return

def convert_to_nifti():
    training_dir = 'E:/baby-seg/training/'
    testing_dir = 'E:/baby-seg/testing/'

    affine = np.eye(4)
    for filename in os.listdir(testing_dir):
        if 'img' in filename:
            basename = filename[:-4]
            data = nib.load(os.path.join(testing_dir, filename)).get_data()

            img = nib.Nifti1Image(data, affine)
            nib.save(img, basename + '.nii.gz')

if __name__ == "__main__":
    make_iseg()
    # convert_to_nifti()