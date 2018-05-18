import SimpleITK as sitk
import numpy as np
from settings import *
import os
from keras import backend as K
from imshow_3D import imshow3D
import copy
import time
import matplotlib.pyplot as plt
import glob
import random


class Helper():
    def __init__(self, s):
        self.s = s
        self.image_spacing_xy = -1

    def mm_to_px(self, mm):
        if self.image_spacing_xy == -1:
            raise Exception('Helper.image_spacing was not set yet')

        # print('mm == {}'.format(mm))
        # print('self.image_spacing_xy == {}'.format(self.image_spacing_xy))
        return mm / self.image_spacing_xy

    def set_image_spacing_xy(self, spacing_3d):
        if spacing_3d[0] != spacing_3d[1]:
            raise Exception('Spacing is different in dimension 0 than in dimension 1 ({} != {})'.format(spacing_3d[0],
                                                                                                        spacing_3d[1]))

        self.image_spacing_xy = spacing_3d[0]

    def getImagePaths(self, nrs, get_all):
        x_all_path = []
        y_all_path = []
        la_all_path = []

        for i in nrs:
            x_all_path.append('{0}input/{1}/p{2}/de_{3}_{2}.nrrd'.format(self.s.PATH_TO_DATA, self.s.PRE_OR_POST_NAME,
                                                                         i, self.s.PRE_OR_POST_XX))

            y_all_path.append('{0}annotations/staple_{2}_{1}.gipl'.format(self.s.PATH_TO_DATA, i,
                                                                          self.s.PRE_OR_POST_XX))

            la_all_path.append('{0}input/{3}/p{1}/la_seg_{2}_{1}.nrrd'.format(self.s.PATH_TO_DATA, i,
                                                                             self.s.PRE_OR_POST_XX,
                                                                             self.s.PRE_OR_POST_NAME))
            # x_all_path.append('../data/ctcf/lge_PLDP_rsmp.nii.gz')
            # y_all_path.append('../data/ctcf/la_seg_PLDP.nii.gz')
            # la_all_path.append('../data/ctcf/la_seg_PLDP.nii.gz')

        if get_all:
            return x_all_path, y_all_path, la_all_path
        elif self.s.GROUND_TRUTH == 'scar_fibrosis':
            return x_all_path, y_all_path
        elif self.s.GROUND_TRUTH == 'left_atrium':
            return x_all_path, la_all_path

    def loadImages(self, pathNames):
        im_out = []
        for p in pathNames:
            im = sitk.GetArrayFromImage(sitk.ReadImage(p))
            im_out.append(im)

        return im_out

    def loadImageSpacing(self, pathNames):
        spacing = []
        for p in pathNames:
            spacing.append(
                sitk.ReadImage(p).GetSpacing()
            )

        return spacing

    def cropImage(self, I, corner, dims):
        c = copy.copy(corner)
        for i in range(3):
            if c[i] < 0:
                pad_elem = [(0, 0), (0, 0), (0, 0)]
                pad_elem[i] = (-c[i], dims[i] + c[i] - I.shape[i])
                pad_elem = tuple(pad_elem)
                I = np.pad(I, pad_elem, 'constant', constant_values=0)
                c[i] = 0

        d, h, w = dims
        z, y, x = c
        return I[z:z+d, y:y+h, x:x+w]

    def getModelResultsPath(self, model_name):
        model_folder = '{}{}/'.format(self.s.PATH_TO_MODELS, model_name)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        return model_folder

    def getModelPredictPath(self, model_name):
        predict_folder = '{}predict/'.format(self.getModelResultsPath(model_name))
        if not os.path.exists(predict_folder):
            os.makedirs(predict_folder)
        return predict_folder

    def getModelPath(self, model_name):
        return '{}model.h5'.format(self.getModelResultsPath(model_name))

    def getModelTrainingTimePath(self, model_name):
        return '{}training_duration.txt'.format(self.getModelResultsPath(model_name))

    def getModelSettingsPath(self, model_name):
        return '{}settings.py'.format(self.getModelResultsPath(model_name))

    def getLogPath(self, model_name):
        return "{}log.p".format(self.getModelResultsPath(model_name))

    def getAugPath(self):
        path = self.s.PATH_TO_AUG
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def getArtPath(self):
        path = self.s.PATH_TO_ART
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def getAugImagesPath(self, img_nr, aug_nr, z, get_all):
        aug_path = self.getAugPath()

        x_folder = '{}input/{}/p{}/'.format(aug_path, self.s.PRE_OR_POST_NAME, img_nr)
        y_folder = '{}annotations/'.format(aug_path)
        la_folder = '{}input/{}/p{}/'.format(aug_path, self.s.PRE_OR_POST_NAME, img_nr)

        if not os.path.exists(x_folder):
            os.makedirs(x_folder)
        if not os.path.exists(y_folder):
            os.makedirs(y_folder)
        if not os.path.exists(la_folder):
            os.makedirs(la_folder)

        x_path = '{}de_{}_{}_{}_{}.nii.gz'.format(x_folder, self.s.PRE_OR_POST_XX, img_nr, z, aug_nr)
        y_path = '{}staple_{}_{}_{}_{}.nii.gz'.format(y_folder, self.s.PRE_OR_POST_XX, img_nr, z, aug_nr)
        la_path = '{}la_seg_{}_{}_{}_{}.nii.gz'.format(la_folder, self.s.PRE_OR_POST_XX, img_nr, z, aug_nr)

        if get_all:
            return x_path, y_path, la_path
        elif self.s.GROUND_TRUTH == 'left_atrium':
            return x_path, la_path
        elif self.s.GROUND_TRUTH == 'scar_fibrosis':
            return x_path, y_path

    def getRandomArtificialPositiveImagePath(self, get_all):
        aug_path = self.getArtPath()

        x_folder = '{}input/'.format(aug_path)
        y_folder = '{}annotations/'.format(aug_path)
        la_folder = '{}input/'.format(aug_path)

        x_path = random.choice(
            glob.glob('{}de_*.nii.gz'.format(x_folder))
        ).replace('\\', '/')

        y_path = x_path.replace(x_folder, y_folder)
        y_path = y_path.replace('de_', 'staple_')
        la_path = x_path.replace(x_folder, la_folder)
        la_path = la_path.replace('de_', 'la_seg_')

        if get_all:
            return x_path, y_path, la_path
        elif self.s.GROUND_TRUTH == 'left_atrium':
            return x_path, la_path
        elif self.s.GROUND_TRUTH == 'scar_fibrosis':
            return x_path, y_path

    def getArtImagesPath(self, img_nr, art_nr, z, get_all):
        aug_path = self.getArtPath()

        x_folder = '{}input/'.format(aug_path)
        y_folder = '{}annotations/'.format(aug_path)
        la_folder = '{}input/'.format(aug_path)

        if not os.path.exists(x_folder):
            os.makedirs(x_folder)
        if not os.path.exists(y_folder):
            os.makedirs(y_folder)
        if not os.path.exists(la_folder):
            os.makedirs(la_folder)

        x_path = '{}de_{}_{}_{}.nii.gz'.format(x_folder, img_nr, z, art_nr)
        y_path = '{}staple_{}_{}_{}.nii.gz'.format(y_folder, img_nr, z, art_nr)
        la_path = '{}la_seg_{}_{}_{}.nii.gz'.format(la_folder, img_nr, z, art_nr)

        if get_all:
            return x_path, y_path, la_path
        elif self.s.GROUND_TRUTH == 'left_atrium':
            return x_path, la_path
        elif self.s.GROUND_TRUTH == 'scar_fibrosis':
            return x_path, y_path

    def getBOPath(self, model_name):
        return "{}bo.p".format(self.getModelResultsPath(model_name))

    def getNrStepsPath(self, model_name):
        return "{}nr_steps.p".format(self.getModelResultsPath(model_name))

    def getClassWeightAuto(self, y_patches):
        n_pos = np.sum(y_patches)
        n_neg = np.sum((y_patches == 0).astype(int))

        return n_neg / n_pos

    def getNoScarPaths(self, nrs):
        no_scar_paths = []
        la_seg_paths = []
        sf_seg_paths = []
        for i in nrs:
            p_folder = self.s.PATH_TO_NO_SCAR_PRE + 'p{}/'.format(i)
            sf_folder = '{}annotations/'.format(self.s.PATH_TO_DATA)

            no_scar_paths.append(p_folder + 'de_a_{}.nrrd'.format(i))
            la_seg_paths.append(p_folder + 'la_seg_a_{}.nrrd'.format(i))
            sf_seg_paths.append(sf_folder + 'staple_a_{}.gipl'.format(i))

        for i in nrs:
            p_folder = self.s.PATH_TO_NO_SCAR_POST + 'p{}/'.format(i)
            sf_folder = '{}annotations/'.format(self.s.PATH_TO_DATA)

            no_scar_paths.append(p_folder + 'de_b_{}.nrrd'.format(i))
            la_seg_paths.append(p_folder + 'la_seg_b_{}.nrrd'.format(i))
            sf_seg_paths.append(sf_folder + 'staple_b_{}.gipl'.format(i))
        return no_scar_paths, la_seg_paths, sf_seg_paths

    # Thanks to https://github.com/keras-team/keras/issues/3611
    def dice_coef(self, y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return -self.dice_coef(y_true, y_pred)

    def weighted_binary_cross_entropy(self, y_true, y_pred):
        y_pred_bw = K.round(y_pred)
        m = y_true - (y_true == y_pred_bw)
        FNentropy = K.binary_crossentropy(m * y_pred, m * y_true)

        m = y_pred - (y_pred_bw == y_pred)
        FPentropy = K.binary_crossentropy(m * y_pred, m * y_true)

        m = y_pred_bw == y_pred
        TPentropy = K.binary_crossentropy(m * y_pred, m * y_true)

        m = y_pred + y_true - (y_pred_bw == y_pred)
        TNentropy = K.binary_crossentropy(m * y_pred, m * y_true)
        print("FN_CLASS_WEIGHT == {}".format(self.s.FN_CLASS_WEIGHT))
        return FNentropy * self.s.FN_CLASS_WEIGHT + FPentropy + TPentropy + TNentropy

    def custom_loss(self, y_true, y_pred):
        if self.s.LOSS_FUNCTION == 'weighted_binary_cross_entropy':
            return self.weighted_binary_cross_entropy(y_true, y_pred)
        elif self.s.LOSS_FUNCTION == 'dice':
            return self.dice_coef_loss(y_true, y_pred)
        else:
            raise Exception('Loss function {} is not (yet) supported.'.format(self.s.LOSS_FUNCTION))

    def imshow(self, img):
        plt.figure()
        plt.imshow(img, cmap='Greys_r')
        plt.show()
