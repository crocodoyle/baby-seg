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
    ibis_dir = 'E:/IBIS_iSeg/'
    scratch_dir = 'E:/'
    numImgs = 23 + 50

    x_dim = 144
    y_dim = 192
    z_dim = 256

    f = h5py.File(scratch_dir + 'baby-seg.hdf5', 'w')
    f.create_dataset('images', (numImgs, x_dim, y_dim, z_dim, 2), dtype='float16')
    f.create_dataset('labels', (numImgs, x_dim, y_dim, z_dim, 1), dtype='uint8')

    numSeen = {}
    numSeen['T1'] = 0
    numSeen['T2'] = 0
    numSeen['labels'] = 0

    maxVal = {}
    maxVal['T1'] = 0
    maxVal['T2'] = 0

    for filenum in range(1, 11):

        t1 = 'subject-' + str(filenum) + '-T1.img'
        t2 = 'subject-' + str(filenum) + '-T2.img'
        label = 'subject-' + str(filenum) + '-label.img'

        print('subject', filenum)

        i = int(filenum) - 1

        f['images'][i, ..., 0] = nib.load(os.path.join(training_dir, t1)).get_data()[:, :, :, 0]
        f['images'][i, ..., 1] = nib.load(os.path.join(training_dir, t2)).get_data()[:, :, :, 0]
        f['labels'][i, ...] = nib.load(os.path.join(training_dir, label)).get_data()

    for filenum in range(11, 24):

        t1 = 'subject-' + str(filenum) + '-T1.img'
        t2 = 'subject-' + str(filenum) + '-T2.img'

        print('subject', filenum)

        i = int(filenum) - 1

        if '23' in str(filenum):
            f['images'][i, ..., 0] = nib.load(os.path.join(testing_dir, t1)).get_data()[8:-8, :, :, 0]
            f['images'][i, ..., 1] = nib.load(os.path.join(testing_dir, t2)).get_data()[8:-8, :, :, 0]
        else:
            f['images'][i, ..., 0] = nib.load(os.path.join(testing_dir, t1)).get_data()[:, :, :, 0]
            f['images'][i, ..., 1] = nib.load(os.path.join(testing_dir, t2)).get_data()[:, :, :, 0]

    i = 23
    for filename in os.listdir(ibis_dir):

        if 'T1' in filename:
            id = filename.split('_')[0]

            print('IBIS id', id, int(i))

            t1 = id + '_V06_T1.nii.gz'
            t2 = id + '_V06_T2.nii.gz'
            label = id + '_V06_label.nii.gz'

            f['images'][i, ..., 0] = nib.load(os.path.join(ibis_dir, t1)).get_data()
            f['images'][i, ..., 1] = nib.load(os.path.join(ibis_dir, t2)).get_data()
            f['labels'][i, ..., 0] = nib.load(os.path.join(ibis_dir, label)).get_data()

            i += 1

    for i in range(numImgs):
        max_t1 = np.max(f['images'][i, ..., 0])
        max_t2 = np.max(f['images'][i, ..., 1])
        print(max_t1, max_t2)
        f['images'][i, ..., 0] = np.divide(f['images'][i, ..., 0], max_t1)
        f['images'][i, ..., 1] = np.divide(f['images'][i, ..., 1], max_t2)

    #     max_t1 = np.max(t1)
    #     max_t2 = np.max(t2)
    #
    #     if max_t1 > maxVal['T1']:
    #         maxVal['T1'] = max_t1
    #     if max_t2 > maxVal['T2']:
    #         maxVal['T2'] = max_t2
    #
    # print(maxVal)
    # for i in range(numImgs):
    #     f['images'][i, ..., 0] = np.divide(f['images'][i, ..., 0], maxVal['T1'])
    #     f['images'][i, ..., 1] = np.divide(f['images'][i, ..., 1], maxVal['T2'])
    #     # f['images'][i, ..., 2] = np.divide(f['images'][i, ..., 0], f['images'][i, ..., 1])

    # pretend background is CSF for easier training
    for i in range(numImgs):
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