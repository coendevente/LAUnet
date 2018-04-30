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


def buildUNet():
    """
    Import Unet from unet_3D
    :return: model from Unet
    """
    model = UNet(PATCH_SIZE + (1, ), dropout=.5, batchnorm=True, depth=3)

    model.compile(optimizer=Adam(lr=1e-5), loss=custom_loss, metrics=['binary_accuracy'])

    return model


def getRandomPatch(x_full, y_full):
    if x_full.shape != y_full.shape:
        raise Exception('x_full.shape != y_full.shape ({} != {})'.format(x_full.shape, y_full.shape))

    z_corner = random.randint(0, x_full.shape[0] - PATCH_SIZE[0])
    y_corner = random.randint(0, x_full.shape[1] - PATCH_SIZE[1])
    x_corner = random.randint(0, x_full.shape[2] - PATCH_SIZE[2])
    corner = (z_corner, y_corner, x_corner)

    x_patch = cropImage(x_full, corner, PATCH_SIZE)
    y_patch = cropImage(y_full, corner, PATCH_SIZE)

    return x_patch, y_patch


def getRandomPatches(x_full, y_full, nr):
    x = []
    y = []
    for j in range(nr):
        positive_batch = random.random() < POS_NEG_PATCH_PROP  # Whether batch should be positive

        its = 0
        while its == 0 or (np.sum(y_j) > 0 and not positive_batch) \
                or (np.sum(y_j) == 0 and positive_batch):
            its += 1

            # Pick current image randomly
            i_nr = random.randint(0, len(x_full) - 1)
            x_full_i, y_full_i = augment(x_full[i_nr], y_full[i_nr])

            x_j, y_j = getRandomPatch(x_full_i, y_full_i)

        print('Found {} batch after {} iterations'.format(bool(positive_batch), its))

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