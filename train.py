from settings import *
import numpy as np
import h5py
import SimpleITK as sitk
from itertools import chain
import matplotlib.pyplot as plt

from helper_functions import *
from unet import UNet

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.optimizers import *

from augment import augment
from offline_augment import offline_augment

import random
import time

import pickle

from imshow_3D import imshow3D

from shutil import copyfile

sliceInformation = {}

def buildUNet():
    """
    Import Unet from unet_3D
    :return: model from Unet
    """
    ps = PATCH_SIZE
    if NR_DIM == 2:
        ps = ps[1:]
    model = UNet(ps + (1, ), dropout=.5, batchnorm=True, depth=3)

    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=custom_loss, metrics=['binary_accuracy'])

    return model


def getRandomPatch(x_full, y_full, set_idx):
    i = random.randint(0, len(x_full) - 1)

    if AUGMENT_ONLINE:
        x = x_full[i]
        y = y_full[i]
    else:
        zl = sliceInformation[set_idx[i]].shape[0]
        s_nr = random.randint(0, zl - 1 - PATCH_SIZE[0])
        x, y = offline_augment(set_idx[i], range(s_nr, s_nr + PATCH_SIZE[0]))

    z_corner = random.randint(0, x.shape[0] - PATCH_SIZE[0])
    y_corner = random.randint(0, x.shape[1] - PATCH_SIZE[1])
    x_corner = random.randint(0, x.shape[2] - PATCH_SIZE[2])
    corner = [z_corner, y_corner, x_corner]

    if AUGMENT_ONLINE:
        x, y = augment(x[z_corner:z_corner + PATCH_SIZE[0]], y[z_corner:z_corner + PATCH_SIZE[0]], False)
        corner[0] = 0

    x_patch = cropImage(x, corner, PATCH_SIZE)
    y_patch = cropImage(y, corner, PATCH_SIZE)

    return x_patch, y_patch


def getRandomPositiveImage(x_full, y_full, set_idx):
    i = random.randint(0, len(x_full) - 1)

    x_pos, y_pos = x_full[i], y_full[i]

    if np.sum(y_pos) == 0:
        x_pos, y_pos = getRandomPositiveImage(x_full, y_full, set_idx)

    return x_pos, y_pos


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

    # print("len(nz) == {}".format(len(nz)))
    # print("nz[0].shape == {}".format(nz[0].shape))
    # print("nz[1].shape == {}".format(nz[1].shape))
    # print("nz[2].shape == {}".format(nz[2].shape))

    nz_i = random.randint(0, nz[0].shape[0] - 1)
    nz_yx = (nz[1][nz_i], nz[2][nz_i])
    ranges = ([-PATCH_SIZE[1] + 1, 0], [-PATCH_SIZE[2] + 1, 0])

    for i in range(2):
        if nz_yx[i] - PATCH_SIZE[i + 1] < 0:
            ranges[i][0] = -nz_yx[i]
        if nz_yx[i] + PATCH_SIZE[i + 1] > x.shape[i + 1]:
            ranges[i][1] = - (PATCH_SIZE[i + 1] - (x.shape[i + 1] - nz_yx[i]))

    # print("x.shape == {}".format(x.shape))
    # print("nz_yx == {}".format(nz_yx))
    # print("ranges == {}".format(ranges))

    z_corner = 0
    y_corner = nz_yx[0] + random.randint(ranges[0][0], ranges[0][1])
    x_corner = nz_yx[1] + random.randint(ranges[1][0], ranges[1][1])
    corner = (z_corner, y_corner, x_corner)

    x_patch = cropImage(x, corner, PATCH_SIZE)
    y_patch = cropImage(y, corner, PATCH_SIZE)

    return x_patch, y_patch, True


def getRandomPositiveSlicesOffline(set_idx):
    its = 0
    while its == 0 or s_nr + PATCH_SIZE[0] > sliceInformation[img_nr].shape[0]:
        its += 1
        img_nr = random.choice(set_idx)
        w = np.where(sliceInformation[img_nr])
        # print("w == {}".format(w))
        s_nr = np.random.choice(w[0])

    x_s, y_s = offline_augment(img_nr, range(s_nr, s_nr + PATCH_SIZE[0]))

    return x_s, y_s


def getRandomPositivePatch(x_full, y_full, set_idx):
    if AUGMENT_ONLINE:
        x_i, y_i = getRandomPositiveImage(x_full, y_full)
        x_s, y_s = getRandomPositiveSlices(x_i, y_i)
        x_s, y_s = augment(x_s, y_s, False)
    else:
        x_s, y_s = getRandomPositiveSlicesOffline(set_idx)

    x_patch, y_patch, found = getRandomPositivePatchAllSlices(x_s, y_s)

    if not found:
        x_patch, y_patch = getRandomPositivePatch(x_full, y_full, set_idx)

    return x_patch, y_patch


def getRandomPatches(x_full, y_full, nr, set_idx):
    x = []
    y = []
    for j in range(nr):
        positive_patch = random.random() < POS_NEG_PATCH_PROP  # Whether batch should be positive

        if not positive_patch:
            x_j, y_j = getRandomPatch(x_full, y_full, set_idx)
        else:
            x_j, y_j = getRandomPositivePatch(x_full, y_full, set_idx)

        # print(positive_patch)
        # imshow3D(np.concatenate((x_j / np.max(x_j), y_j), axis=2))

        x.append(x_j)
        y.append(y_j)
    x = np.array(x)
    y = np.array(y)

    sh = x.shape
    if NR_DIM == 2:
        sh = sh[:1] + sh[2:]

    x = np.reshape(x, sh + (1, ))
    y = np.reshape(y, sh + (1, ))
    return x, y


def updateSliceInformation(y_all, set_idx):
    for i in range(len(set_idx)):
        sliceInformation[set_idx[i]] = []
        for z in range(y_all[i].shape[0]):
            pos = np.sum(y_all[i][z]) > 0
            sliceInformation[set_idx[i]].append(pos)
        sliceInformation[set_idx[i]] = np.array(sliceInformation[set_idx[i]])


def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

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

    updateSliceInformation(y_full_train, TRAINING_SET)
    updateSliceInformation(y_full_val, VALIDATION_SET)
    print("sliceInformation == {}".format(sliceInformation))

    print('len(x_full_train) == {}'.format(len(x_full_train)))

    if FN_CLASS_WEIGHT_SETTINGS == 'auto':
        _, y_patches = getRandomPatches(x_full_train + x_full_val, y_full_train + y_full_val, AUTO_CLASS_WEIGHT_N,
                                        TRAINING_SET + VALIDATION_SET)
        w = getClassWeightAuto(y_patches)
        print("w == {}".format(w))
        set_fn_class_weight(w)

    model = buildUNet()

    log = {'training': {'loss': [], 'accuracy': []}, 'validation': {'loss': [], 'accuracy': []}}

    start_time = time.time()
    lowest_val_loss = float("inf")
    lowest_train_loss = float("inf")

    copyfile('settings.py', getModelSettingsPath(MODEL_NAME))

    print("Start training...")

    log_path = getLogPath(MODEL_NAME)
    log['fn_class_weight'] = get_fn_class_weight()

    for i in range(NR_BATCHES):

        print('{}s passed. Starting getRandomPatches.'.format(round(time.time() - start_time)))
        x_train, y_train = getRandomPatches(x_full_train, y_full_train, BATCH_SIZE, TRAINING_SET)
        x_val, y_val = getRandomPatches(x_full_val, y_full_val, NR_VAL_PATCH_PER_ITER, VALIDATION_SET)
        print('{}s passed. Ended getRandomPatches.'.format(round(time.time() - start_time)))

        train_loss = model.train_on_batch(x_train, y_train)
        log['training']['loss'].append(train_loss[0])
        log['training']['accuracy'].append(train_loss[1])

        val_loss = model.test_on_batch(x_val, y_val)
        log['validation']['loss'].append(val_loss[0])
        log['validation']['accuracy'].append(val_loss[1])
        pickle.dump(log, open(log_path, "wb"))

        if lowest_val_loss > val_loss[0]:
            lowest_val_loss = val_loss[0]
            model_path = getModelPath(MODEL_NAME)
            model.save(model_path)

        if lowest_train_loss > train_loss[0]:
            lowest_train_loss = train_loss[0]

        ETA = round(time.time() - start_time) * (1/((i + 1) / NR_BATCHES) - 1)
        # ETA = 0
        print(('{}s passed. ETA is {}s. Finished training on batch {}/{} ({}%). Latest, lowest validation loss: {}, {}.' +
              ' Latest, lowest training loss: {}, {}.').format(
            round(time.time() - start_time), ETA, i + 1, NR_BATCHES, (i + 1) / NR_BATCHES * 100, val_loss[0],
            lowest_val_loss, train_loss[0], lowest_train_loss))

    training_duration = round(time.time() - start_time)
    print('Training took {} seconds.'.format(training_duration))
    time_file = open(getModelTrainingTimePath(MODEL_NAME), "w")
    time_file.write('Training took {}'.format(training_duration))
    time_file.close()


if __name__ == "__main__":
    main()