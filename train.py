import numpy as np
import h5py
import SimpleITK as sitk
from itertools import chain
import matplotlib.pyplot as plt

from settings import *
from helper_functions import *
from unet_3D import UNet

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.optimizers import *

from augment import augment

import random
import time

import pickle

from imshow_3D import imshow3D


def buildUNet():
    """
    Import Unet from unet_3D
    :return: model from Unet
    """
    model = UNet(PATCH_SIZE + (1, ), dropout=.5, batchnorm=True, depth=3)

    model.compile(optimizer=Adam(lr=1e-5), loss=custom_loss, metrics=['binary_accuracy'])

    return model


def getRandomPatch(x_full, y_full):
    i = random.randint(0, len(x_full) - 1)
    x = x_full[i]
    y = y_full[i]

    z_corner = random.randint(0, x.shape[0] - PATCH_SIZE[0])
    y_corner = random.randint(0, x.shape[1] - PATCH_SIZE[1])
    x_corner = random.randint(0, x.shape[2] - PATCH_SIZE[2])
    corner = (z_corner, y_corner, x_corner)

    x_patch = cropImage(x, corner, PATCH_SIZE)
    y_patch = cropImage(y, corner, PATCH_SIZE)

    return x_patch, y_patch


def getRandomPositiveImage(x_full, y_full):
    i = random.randint(0, len(x_full) - 1)

    if np.sum(y_full[i]) == 0:
        y_full = getRandomPositiveImage(x_full, y_full)

    return x_full[i], y_full[i]


def getRandomPositiveSlices(x_i, y_i):
    its = 0
    while its == 0 or nz_z + PATCH_SIZE[0] > x_i.shape[0]:
        its += 1
        nz = np.nonzero(y_i)
        nz_z = random.choice(nz[0])

    x_s = x_i[nz_z:nz_z + x_i.shape[0]]
    y_s = y_i[nz_z:nz_z + y_i.shape[0]]
    return x_s, y_s


def getRandomPositivePatchAllSlices(x, y):
    if np.sum(y) == 0:
        return 0, 0, False

    nz = np.nonzero(y)
    nz_i = random.randint(0, nz[0].shape[0])
    nz_yx = (nz[1][nz_i], nz[2][nz_i])
    ranges = ([-PATCH_SIZE[1] + 1, PATCH_SIZE[1] - 1], [-PATCH_SIZE[2] + 1, PATCH_SIZE[2] - 1])

    for i in range(2):
        if nz_yx[i] - PATCH_SIZE[i + 1] + 1 < 0:
            ranges[i][0] = nz_yx[i]
        if nz_yx[i] + PATCH_SIZE[i + 1] - 1 >= x.shape[i + 1]:
            ranges[i][0] = x.shape[i + 1] - nz_yx[i] - 1

    z_corner = 0
    y_corner = nz_yx[0] + random.randint(ranges[0][0], ranges[0][1])
    x_corner = nz_yx[1] + random.randint(ranges[1][0], ranges[1][1])
    corner = (z_corner, y_corner, x_corner)

    x_patch = cropImage(x, corner, PATCH_SIZE)
    y_patch = cropImage(y, corner, PATCH_SIZE)

    return x_patch, y_patch, True


def getRandomPositivePatch(x_full, y_full):
    x_i, y_i = getRandomPositiveImage(x_full, y_full)
    x_s, y_s = getRandomPositiveSlices(x_i, y_i)
    x_aug, y_aug = augment(x_s, y_s)
    x_patch, y_patch, found = getRandomPositivePatchAllSlices(x_aug, y_aug)

    if not found:
        x_patch, y_patch = getRandomPositivePatch(x_full, y_full)

    return x_patch, y_patch


def getRandomPatches(x_full, y_full, nr):
    x = []
    y = []
    for j in range(nr):
        positive_patch = random.random() < POS_NEG_PATCH_PROP  # Whether batch should be positive

        if not positive_patch:
            x_j, y_j = getRandomPatch(x_full, y_full)
        else:
            x_j, y_j = getRandomPositivePatch(x_full, y_full)
            # imshow3D(np.concatenate((x_j / np.max(x_j), y_j), axis=2))

        x.append(x_j)
        y.append(y_j)
    x = np.array(x)
    y = np.array(y)

    x = np.reshape(x, x.shape + (1, ))
    y = np.reshape(y, y.shape + (1, ))
    return x, y


def main():
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    print(device_lib.list_local_devices())

    x_all_path, y_all_path = getImagePaths(range(1, 31))

    # Full images
    x_full_all = loadImages(x_all_path)
    y_full_all = loadImages(y_all_path)

    # Divide full images in training and validation
    x_full_train = [x_full_all[i - 1] for i in TRAINING_SET]
    y_full_train = [y_full_all[i - 1] for i in TRAINING_SET]
    x_full_val = [x_full_all[i - 1] for i in VALIDATION_SET]
    y_full_val = [y_full_all[i - 1] for i in VALIDATION_SET]

    print('len(x_full_train) == {}'.format(len(x_full_train)))

    model = buildUNet()

    loss = {'training': {'loss': [], 'accuracy': []}, 'validation': {'loss': [], 'accuracy': []}}

    start_time = time.time()
    lowest_val_loss = float("inf")
    lowest_train_loss = float("inf")

    print("Start training...")
    for i in range(NR_BATCHES):

        x_train, y_train = getRandomPatches(x_full_train, y_full_train, BATCH_SIZE)
        x_val, y_val = getRandomPatches(x_full_val, y_full_val, NR_VAL_PATCH_PER_ITER)

        train_loss = model.train_on_batch(x_train, y_train)
        loss['training']['loss'].append(train_loss[0])
        loss['training']['accuracy'].append(train_loss[1])

        val_loss = model.test_on_batch(x_val, y_val)
        loss['validation']['loss'].append(val_loss[0])
        loss['validation']['accuracy'].append(val_loss[1])

        loss_path = getLossPath(MODEL_NAME)
        pickle.dump(loss, open(loss_path, "wb"))

        if lowest_val_loss > val_loss[0]:
            lowest_val_loss = val_loss[0]
            model_path = getModelPath(MODEL_NAME)
            model.save(model_path)

        if lowest_train_loss > train_loss[0]:
            lowest_train_loss = train_loss[0]

        print(('{}s passed. Start training on batch {}/{} ({}%). Latest, lowest training loss: {}, {}.' +
              ' Latest, lowest validation loss: {}, {}.').format(
            round(time.time() - start_time), i + 1, NR_BATCHES, (i + 1) / NR_BATCHES * 100, val_loss[0],
            lowest_val_loss, train_loss[0], lowest_train_loss))

    print('Training took {} seconds.'.format((round(time.time() - start_time))))


if __name__ == "__main__":
    main()