import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import nibabel as nib
from keras.utils import to_categorical


# Filter and keep only slices that have at least some segmentation mask

def filter_slices(nifti_file, mask_file):
    filtered_slices = []
    filtered_segmentation = []

    # Load NIfTI files
    nifti_img = nib.load(nifti_file).get_fdata()
    mask_img = nib.load(mask_file).get_fdata()

    # Going through each slice to filter out useful ones
    for i in range(nifti_img.shape[-1]):
        # Check if both CT scan slice and segmentation mask slice contain non-zero values
        if np.any(nifti_img[..., i] != 0) and np.any(mask_img[..., i] != 0):
            filtered_slices.append(nifti_img[..., i])
            filtered_segmentation.append(mask_img[..., i])

    return (filtered_slices, filtered_segmentation)


def read_and_filter_nifti_file(img_dir, img_list, masks_dir, masks_list):
    slices_array = []
    masks_array = []

    for i in range(len(img_list)):
        slice_path = img_dir + '\\' + img_list[i]
        mask_path = masks_dir + '\\' + masks_list[i]

        print(slice_path)
        print(mask_path)

        # Filter out the slices
        slices, masks = filter_slices(slice_path, mask_path)

        # Creating a list of patient slices
        slices_array.append(slices)
        masks_array.append(masks)

    # Stacking slices
    slices_array = np.concatenate(slices_array, axis=0)
    masks_array = np.concatenate(masks_array, axis=0)

    return (slices_array, masks_array)


def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)

    # keras needs the generator infinite, so we will use while true
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            X, Y = read_and_filter_nifti_file(img_dir, img_list[batch_start:limit], mask_dir,
                                              mask_list[batch_start:limit])
            # X=np.transpose(X,(1,2,0))
            # Y=np.transpose(Y,(1,2,0))
            Y = to_categorical(Y, 3)
            X = np.expand_dims(X, axis=-1)
            # Y=np.expand_dims(Y, axis=-1)
            print('X shape: ', X.shape)
            print('Y shape: ', Y.shape)
            yield (X, Y)  # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size
