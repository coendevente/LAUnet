from settings import *
import numpy as np
import h5py
import SimpleITK as sitk
from itertools import chain
import matplotlib.pyplot as plt

from helper_functions import *
from unet import UNet

import os
os.environ["PATH"] += os.pathsep + 'C:/Anaconda3/Library/bin/graphviz/'

from tensorflow.python.client import device_lib
from keras.optimizers import *
from keras.utils import plot_model
import tensorflow as tf

from online_augment import OnlineAugmenter
from offline_augment import OfflineAugmenter

import random
import time

import pickle

from imshow_3D import imshow3D

from shutil import copyfile

from skimage import img_as_bool

import multiprocessing
from joblib import Parallel, delayed


class Train:
    def __init__(self, s, h):
        self.sliceInformation = {}
        self.s = s
        self.h = h
        self.offline_augmenter = OfflineAugmenter(s, h)
        self.online_augmenter = OnlineAugmenter(s, h)

    def buildUNet(self):
        """
        Import Unet from unet_3D
        :return: model from Unet
        """
        ps = self.s.PATCH_SIZE
        if self.s.VARIABLE_PATCH_SIZE:
            ps = (None, None, None)
        if self.s.NR_DIM == 2:
            ps = ps[1:]
        model = UNet(ps + (1, ), self.s.NR_DIM, dropout=self.s.DROPOUT, batchnorm=True, depth=self.s.UNET_DEPTH,
                     doeverylevel=self.s.DROPOUT_AT_EVERY_LEVEL, inc_rate=self.s.FEATURE_MAP_INC_RATE,
                     aux_loss=self.s.USE_ANY_SCAR_AUX)

        if self.s.USE_ANY_SCAR_AUX:
            model.compile(optimizer=Adam(lr=self.s.LEARNING_RATE), loss={'main_output': self.h.custom_loss,
                                                                         'aux_output': 'mean_squared_error'},
                          metrics=['binary_accuracy'], loss_weights={'main_output': self.s.MAIN_OUTPUT_LOSS_WEIGHT,
                                                                     'aux_output': self.s.AUX_OUTPUT_LOSS_WEIGHT})
        else:
            model.compile(optimizer=Adam(lr=self.s.LEARNING_RATE), loss=self.h.custom_loss,
                          metrics=['binary_accuracy'])

        return model

    def getRandomNegativePatch(self, x_full, y_full, set_idx):
        i = random.randint(0, len(x_full) - 1)

        lap = -1

        if self.s.AUGMENT_ONLINE:
            x = x_full[i]
            y = y_full[i]
        else:
            zl = self.sliceInformation[set_idx[i]].shape[0]

            first = True
            while first or self.sliceInformation[set_idx[i]][s_nr]:
                first = False
                s_nr = random.randint(0, zl - 1 - self.s.PATCH_SIZE[0])

            x, y, _, lap = self.offline_augmenter.offline_augment(set_idx[i], range(s_nr, s_nr + self.s.PATCH_SIZE[0]),
                                                                  True)

        corner = [0, 0, 0]

        for i in range(3):
            if x.shape[i] < self.s.PATCH_SIZE[i]:
                corner[i] = round((x.shape[i] - self.s.PATCH_SIZE[i]) / 2)
            else:
                corner[i] = random.randint(0, x.shape[i] - self.s.PATCH_SIZE[i])

        if self.s.AUGMENT_ONLINE:
            x, y = self.online_augmenter.augment(x[corner[0]:corner[0] + self.s.PATCH_SIZE[0]], y[corner[0]:corner[0]
                                                 + self.s.PATCH_SIZE[0]], False, None)
            corner[0] = 0

        # print("x.shape == {}".format(x.shape))
        # print("y.shape == {}".format(y.shape))

        if self.s.VARIABLE_PATCH_SIZE:
            x_patch, y_patch, lap_patch = x, y, lap
        else:
            if self.s.PATCH_SIZE[1] <= x.shape[1]:
                x_patch = self.h.cropImage(x, corner, self.s.PATCH_SIZE)
                lap_patch = self.h.cropImage(lap, corner, self.s.PATCH_SIZE)
                y_patch = self.h.cropImage(y, corner, self.s.PATCH_SIZE)
            else:
                x_patch = self.h.rescaleImage(x[corner[0]:corner[0]+self.s.PATCH_SIZE[0]], self.s.PATCH_SIZE[1:])
                lap_patch = (
                        self.h.rescaleImage(lap[corner[0]:corner[0]+self.s.PATCH_SIZE[0]], self.s.PATCH_SIZE[1:]) > 0
                ).astype(int)
                y_patch = (
                        self.h.rescaleImage(y[corner[0]:corner[0]+self.s.PATCH_SIZE[0]], self.s.PATCH_SIZE[1:]) > 0
                ).astype(int)

            if np.sum(y_patch) > 0:
                print('RETRY NEGATIVE PATCH')
                x_patch, lap_patch, y_patch = self.getRandomNegativePatch(x_full, y_full, set_idx)

        x_patch = self.h.pre_process(x_patch)

        return x_patch, lap_patch, y_patch

    def getRandomPositiveImage(self, x_full, y_full, set_idx):
        i = random.randint(0, len(x_full) - 1)
        x_pos, y_pos = x_full[i], y_full[i]

        if np.sum(y_pos) == 0:
            x_pos, y_pos = self.getRandomPositiveImage(x_full, y_full, set_idx)

        return x_pos, y_pos

    def getRandomPositiveSlices(self, x_i, y_i):
        its = 0
        while its == 0 or nz_z + self.s.PATCH_SIZE[0] > x_i.shape[0]:
            its += 1
            nz = np.nonzero(y_i)
            nz_z = random.choice(nz[0])

        x_s = x_i[nz_z:nz_z + x_i.shape[0]]
        y_s = y_i[nz_z:nz_z + y_i.shape[0]]
        return x_s, y_s

    def getRandomPositivePatchAllSlices(self, x, lap, y):
        if np.sum(y) == 0:
            return 0, 0, False

        if self.s.VARIABLE_PATCH_SIZE:
            x_patch, lap_patch, y_patch = x, lap, y
        else:
            nz = np.nonzero(y)

            nz_i = random.randint(0, nz[0].shape[0] - 1)
            nz_yx = (nz[1][nz_i], nz[2][nz_i])
            ranges = ([-self.s.PATCH_SIZE[1] + 1, 0], [-self.s.PATCH_SIZE[2] + 1, 0])

            for i in range(2):
                if nz_yx[i] - self.s.PATCH_SIZE[i + 1] < 0:
                    ranges[i][0] = -nz_yx[i]
                if nz_yx[i] + self.s.PATCH_SIZE[i + 1] > x.shape[i + 1]:
                    ranges[i][1] = -(self.s.PATCH_SIZE[i + 1] - (x.shape[i + 1] - nz_yx[i]))

            corner = [0, 0, 0]

            for i in range(1, 3):
                if x.shape[i] < self.s.PATCH_SIZE[i]:
                    corner[i] = round((x.shape[i] - self.s.PATCH_SIZE[i]) / 2)
                else:
                    corner[i] = nz_yx[i - 1] + random.randint(ranges[i - 1][0], ranges[i - 1][1])

            if self.s.PATCH_SIZE[1] <= x.shape[1]:
                x_patch = self.h.cropImage(x, corner, self.s.PATCH_SIZE)
                lap_patch = self.h.cropImage(lap, corner, self.s.PATCH_SIZE)
                y_patch = self.h.cropImage(y, corner, self.s.PATCH_SIZE)
            else:
                x_patch = self.h.rescaleImage(x[corner[0]:corner[0]+self.s.PATCH_SIZE[0]], self.s.PATCH_SIZE[1:])
                lap_patch = (
                        self.h.rescaleImage(lap[corner[0]:corner[0]+self.s.PATCH_SIZE[0]], self.s.PATCH_SIZE[1:]) > 0
                ).astype(int)
                y_patch = (
                        self.h.rescaleImage(y[corner[0]:corner[0]+self.s.PATCH_SIZE[0]], self.s.PATCH_SIZE[1:]) > 0
                ).astype(int)

        # imshow3D(y_patch)
        # imshow3D(y[corner[0]:corner[0]+self.s.PATCH_SIZE[0]])

        return x_patch, lap_patch, y_patch, True

    def getRandomPositiveSlicesOffline(self, set_idx):
        lap_s = -1

        if random.random() < self.s.ART_FRACTION and set(set_idx) != set(self.s.VALIDATION_SET):
            x_s_path, y_s_path = self.h.getRandomArtificialPositiveImagePath(False, set_idx)

            # print('x_pos_path == {}'.format(x_s_path))
            # print('y_pos_path == {}'.format(y_s_path))

            x_s = self.h.loadImages([x_s_path])[0]
            y_s = self.h.loadImages([y_s_path])[0]

            x_s = np.reshape(x_s, (1,) + x_s.shape)
            y_s = np.reshape(y_s, (1,) + y_s.shape)
        else:
            its = 0
            while its == 0 or s_nr + self.s.PATCH_SIZE[0] > self.sliceInformation[img_nr].shape[0]:
                its += 1
                img_nr = random.choice(set_idx)
                w = np.where(self.sliceInformation[img_nr])
                s_nr = np.random.choice(w[0])

            x_s, y_s, _, lap_s = self.offline_augmenter.offline_augment(img_nr, range(s_nr, s_nr + self.s.PATCH_SIZE[0])
                                                                        , True)
            x_s = self.h.pre_process(x_s)

        return x_s, lap_s, y_s

    def getRandomPositivePatch(self, x_full, y_full, set_idx):
        lap_s = -1

        if self.s.AUGMENT_ONLINE:
            x_i, y_i = self.getRandomPositiveImage(x_full, y_full, set_idx)
            x_s, y_s = self.getRandomPositiveSlices(x_i, y_i)
            x_s, y_s = self.online_augmenter.augment(x_s, y_s, False, None)
            x_s = self.h.pre_process(x_s)
        else:
            x_s, lap_s, y_s = self.getRandomPositiveSlicesOffline(set_idx)

        # imshow3D(
        #     np.concatenate(
        #         (x_s, y_s * np.max(x_s)), axis=2
        #     )
        # )

        x_patch, lap_patch, y_patch, found = self.getRandomPositivePatchAllSlices(x_s, lap_s, y_s)

        if not found:
            x_patch, lap_patch, y_patch = self.getRandomPositivePatch(x_full, y_full, set_idx)

        return x_patch, lap_patch, y_patch

    def getRandomPatches(self, x_full, y_full, nr, set_idx):
        x = []
        y = []
        lap = []

        for j in range(nr):
            positive_patch = random.random() < self.s.POS_NEG_PATCH_PROP  # Whether batch should be positive

            if not positive_patch:
                x_j, lap_j, y_j = self.getRandomNegativePatch(x_full, y_full, set_idx)
            else:
                x_j, lap_j, y_j = self.getRandomPositivePatch(x_full, y_full, set_idx)

            if self.s.VARIABLE_PATCH_SIZE:
                x_j = self.h.resize_to_unet_shape(x_j, self.s.UNET_DEPTH)
                lap_j = self.h.resize_to_unet_shape(lap_j, self.s.UNET_DEPTH)
                y_j = self.h.resize_to_unet_shape(y_j, self.s.UNET_DEPTH)

            # print("positive_patch == {}".format(positive_patch))
            # print("x_j.shape == {}".format(x_j.shape))
            # print("y_j.shape == {}".format(y_j.shape))
            # print("np.sum(y_j) == {}".format(np.sum(y_j)))
            # imshow3D(np.concatenate((x_j / np.max(x_j), y_j), axis=2))

            x.append(x_j)
            y.append(y_j)
            lap.append(lap_j)

        if self.s.VARIABLE_PATCH_SIZE:
            for i in range(len(x)):
                x[i] = np.reshape(x[i], x[i].shape + (1,))
                y[i] = np.reshape(y[i], y[i].shape + (1,))
                lap[i] = np.reshape(lap[i], lap[i].shape + (1,))
        else:
            x = np.array(x)
            y = np.array(y)
            lap = np.array(lap)

            if self.s.USE_LA_INPUT:
                x = np.concatenate((x, lap), axis=self.s.NR_DIM)

            sh = x.shape
            if self.s.NR_DIM == 2:
                sh = sh[:1] + sh[2:]

            if not self.s.USE_LA_INPUT:
                x = np.reshape(x, sh + (1, ))
            y = np.reshape(y, sh + (1, ))
        return x, lap, y

    def updateSliceInformation(self, y_all, set_idx):
        for i in range(len(set_idx)):
            self.sliceInformation[set_idx[i]] = []
            for z in range(y_all[i].shape[0]):
                pos = np.sum(y_all[i][z]) > 0
                self.sliceInformation[set_idx[i]].append(pos)
            self.sliceInformation[set_idx[i]] = np.array(self.sliceInformation[set_idx[i]])

    def get_aux(self, y):
        y_aux = np.array([int(np.sum(y[j]) > 0) for j in range(y.shape[0])])
        y_aux = np.reshape(y_aux, (y_aux.shape[0], 1))

        return y_aux

    def train(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # self.s.FN_CLASS_WEIGHT = 100
        # model = self.buildUNet()
        # plot_model(model, to_file='model.png')

        self.h.s = self.s

        print(device_lib.list_local_devices())

        x_all_path, y_all_path = self.h.getImagePaths(self.s.ALL_NATURAL_SET, False)

        # Full images
        x_full_all = self.h.loadImages(x_all_path)
        y_full_all = self.h.loadImages(y_all_path)

        self.x_full_all = x_full_all
        self.y_full_all = y_full_all

        # Divide full images in training and validation
        # to_subtract = 1 if self.s.DATA_SET == 'original' else 0

        x_full_train = [x_full_all[i - 1] for i in self.s.TRAINING_SET]
        y_full_train = [y_full_all[i - 1] for i in self.s.TRAINING_SET]
        x_full_val = [x_full_all[i - 1] for i in self.s.VALIDATION_SET]
        y_full_val = [y_full_all[i - 1] for i in self.s.VALIDATION_SET]

        self.updateSliceInformation(y_full_train, self.s.TRAINING_SET)
        self.updateSliceInformation(y_full_val, self.s.VALIDATION_SET)

        if self.s.FN_CLASS_WEIGHT == 'auto' and self.s.LOSS_FUNCTION == 'weighted_binary_cross_entropy':
            _, _, y_patches = self.getRandomPatches(x_full_train + x_full_val, y_full_train + y_full_val,
                                                 self.s.AUTO_CLASS_WEIGHT_N, self.s.TRAINING_SET
                                                 + self.s.VALIDATION_SET)
            self.s.FN_CLASS_WEIGHT = self.h.getClassWeightAuto(y_patches)
            self.h.s = self.s

        model = self.buildUNet()

        print(model.summary())

        log = {'training': {}, 'validation': {}}

        for m in model.metrics_names:
            log['training'][m] = []
            log['validation'][m] = []

        # print(model.metrics_names)

        # log = {'training': {'loss': [], 'accuracy': []}, 'validation': {'loss': [], 'accuracy': []}}

        start_time = time.time()
        lowest_val_loss = float("inf")
        lowest_train_loss = float("inf")

        # copyfile('settings.py', self.h.getModelSettingsPath(self.s.MODEL_NAME))

        print("Start training...")

        log_path = self.h.getLogPath(self.s.MODEL_NAME)
        log['fn_class_weight'] = self.s.FN_CLASS_WEIGHT
        log['settings'] = self.s

        es_j = 0  # Counter for early stopping

        log['stopped_early'] = False
        print("self.s.EARLY_STOPPING == {}".format(self.s.EARLY_STOPPING))
        print("self.s.PATIENTCE_ES == {}".format(self.s.PATIENTCE_ES))

        pickle.dump(log, open(log_path, "wb"))
        for i in range(self.s.NR_BATCHES):
            if self.s.EARLY_STOPPING and self.s.PATIENTCE_ES <= es_j:
                print("Stopped early at iteration {}".format(i))
                log['stopped_early'] = True
                break
            es_j += 1

            print('{}s passed. Starting getRandomPatches.'.format(round(time.time() - start_time)))
            x_train, lap_train, y_train = self.getRandomPatches(x_full_train, y_full_train, self.s.BATCH_SIZE,
                                                                self.s.TRAINING_SET)
            x_val, lap_val, y_val = self.getRandomPatches(x_full_val, y_full_val, self.s.NR_VAL_PATCH_PER_ITER,
                                                 self.s.VALIDATION_SET)
            print('{}s passed. Ended getRandomPatches.'.format(round(time.time() - start_time)))

            if self.s.USE_NORMALIZATION:
                x_train = self.h.normalize_multiple_list(x_train) if self.s.VARIABLE_PATCH_SIZE else \
                          self.h.normalize_multiple_ndarray(x_train)
                x_val = self.h.normalize_multiple_list(x_val) if self.s.VARIABLE_PATCH_SIZE else \
                        self.h.normalize_multiple_ndarray(x_val)

            y_train_all = {'main_output': y_train}
            y_val_all = {'main_output': y_val}

            if self.s.USE_ANY_SCAR_AUX:
                y_train_aux = self.get_aux(y_train)
                y_val_aux = self.get_aux(y_val)

                y_train_all['aux_output'] = y_train_aux
                y_val_all['aux_output'] = y_val_aux

            train_loss = model.train_on_batch(x_train, y_train_all) if not self.s.VARIABLE_PATCH_SIZE else \
                model.train_on_batch(x_train[0], {'main_output': y_train_all['main_output'][0]})

            if self.s.VARIABLE_PATCH_SIZE:
                val_losses = []
                for j in range(len(y_val_all)):
                     val_losses.append(model.test_on_batch(x_val[j], {'main_output': y_train_all['main_output'][j]}))
                val_loss = np.mean(val_losses)
            else:
                val_loss = model.test_on_batch(x_val, y_val_all)

            for m in range(len(model.metrics_names)):
                log['training'][model.metrics_names[m]].append(train_loss[m])
                log['validation'][model.metrics_names[m]].append(val_loss[m])

            pickle.dump(log, open(log_path, "wb"))

            if lowest_val_loss > val_loss[0]:
                lowest_val_loss = val_loss[0]
                model_path = self.h.getModelPath(self.s.MODEL_NAME)
                model.save(model_path)
                lowest_val_loss_i = i
                log['lowest_val_loss'] = lowest_val_loss
                log['lowest_val_loss_i'] = lowest_val_loss_i
                es_j = 0

            if lowest_train_loss > train_loss[0]:
                lowest_train_loss = train_loss[0]

            ETA = round(time.time() - start_time) * (1/((i + 1) / self.s.NR_BATCHES) - 1)

            print(('{}s passed. ETA is {}s. Finished training on batch {}/{} ({}%). Latest, lowest validation loss:' +
                  ' {}, {}. Latest, lowest training loss: {}, {}.').format(
                round(time.time() - start_time), ETA, i + 1, self.s.NR_BATCHES, (i + 1) /
                      self.s.NR_BATCHES * 100, val_loss[0],
                lowest_val_loss, train_loss[0], lowest_train_loss))

        training_duration = round(time.time() - start_time)
        print('Training took {} seconds.'.format(training_duration))

        log['training_duration'] = training_duration
        pickle.dump(log, open(log_path, "wb"))


if __name__ == "__main__":
    s = Settings()
    h = Helper(s)
    t = Train(s, h)
    t.train()
