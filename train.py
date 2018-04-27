import numpy as np
import SimpleITK as sitk
from itertools import chain
import matplotlib.pyplot as plt

from settings import *
from helper_functions import *
from unet_3D import UNet

from keras.optimizers import *
from keras import backend as K

from augment import augment

import random


def custom_loss(y_true, y_pred):
    FNentropy = K.binary_crossentropy((y_true - (K.round(y_true) == y_pred)) * y_pred, y_true)
    FPentropy = K.binary_crossentropy((y_pred - (K.round(y_true) == y_pred)) * y_pred, y_true)
    TPentropy = K.binary_crossentropy((K.round(y_true) == y_pred) * y_pred, y_true)
    TNentropy = K.binary_crossentropy((y_pred + y_true - (K.round(y_true) == y_pred)) * y_pred, y_true)
    return FNentropy * FN_CLASS_WEIGHT + FPentropy + TPentropy + TNentropy


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
        print(j)
        positive_batch = random.random() < POS_NEG_PATCH_PROP  # Whether batch should be positive

        first = True
        while first or (np.sum(y_j) > 0 and not positive_batch) \
                or (np.sum(y_j) == 0 and positive_batch):
            first = False
            print("while in {}".format(j))

            # Pick current image randomly
            i_nr = random.randint(0, len(TRAINING_SET) - 1)
            x_full_i = augment(x_full[i_nr])
            y_full_i = augment(y_full[i_nr])

            x_j, y_j = getRandomPatch(x_full_i, y_full_i)

        x.append(x_j)
        y.append(y_j)

    x = np.reshape(np.array(x), x.shape + (1, ))
    y = np.reshape(np.array(y), y.shape + (1, ))
    return x, y


def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    x_all_path = []
    y_all_path = []
    for i in range(1, 31):
        x_all_path.append('{0}input/{1}/p{2}/de_{3}_{2}.nrrd'.format(PATH_TO_DATA, PRE_OR_POST_NAME, i, PRE_OR_POST_XX))
        y_all_path.append('{0}annotations/staple_{2}_{1}.gipl'.format(PATH_TO_DATA, i, PRE_OR_POST_XX))

    # Full images
    x_full_all = loadImages(x_all_path)
    y_full_all = loadImages(y_all_path)

    # Divide full images in training and validation
    x_full_train = [x_full_all[i] for i in TRAINING_SET]
    y_full_train = [y_full_all[i] for i in TRAINING_SET]
    x_full_val = [x_full_all[i] for i in VALIDATION_SET]
    y_full_val = [y_full_all[i] for i in VALIDATION_SET]

    model = buildUNet()

    plt.figure()
    train_loss_fig = plt.subplot(2, 2, 1)
    train_acc_fig = plt.subplot(2, 2, 2)

    val_loss_fig = plt.subplot(2, 2, 3)
    val_acc_fig = plt.subplot(2, 2, 4)

    for i in range(NR_BATCHES):
        print(i)

        x_train, y_train = getRandomPatches(x_full_train, y_full_train, BATCH_SIZE)
        x_val, y_val = getRandomPatches(x_full_val, y_full_val, NR_VAL_PATCH_PER_ITER)

        train_loss = model.train_on_batch(x_train, y_train)
        train_loss_fig.plot(i, train_loss[0])
        train_acc_fig.plot(i, train_loss[1])

        val_loss = model.test_on_batch(x_val, y_val)
        val_loss_fig.plot(i, val_loss[0])
        val_acc_fig.plot(i, val_loss[1])


if __name__ == "__main__":
    main()