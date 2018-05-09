import SimpleITK as sitk
import numpy as np
from settings import *
import os
from keras import backend as K
from imshow_3D import imshow3D
import copy
import time


class Helper():
    def __init__(self, s):
        self.s = s

    def getImagePaths(self, nrs):
        x_all_path = []
        y_all_path = []
        for i in nrs:
            x_all_path.append('{0}input/{1}/p{2}/de_{3}_{2}.nrrd'.format(self.s.PATH_TO_DATA, self.s.PRE_OR_POST_NAME,
                                                                         i, self.s.PRE_OR_POST_XX))
            if self.s.GROUND_TRUTH == 'scar_fibrosis':
                y_all_path.append('{0}annotations/staple_{2}_{1}.gipl'.format(self.s.PATH_TO_DATA, i,
                                                                              self.s.PRE_OR_POST_XX))
            elif self.s.GROUND_TRUTH == 'left_atrium':
                y_all_path.append('{0}input/{3}/p{1}/la_seg_{2}_{1}.nrrd'.format(self.s.PATH_TO_DATA, i,
                                                                                 self.s.PRE_OR_POST_XX,
                                                                                 self.s.PRE_OR_POST_NAME))
        return x_all_path, y_all_path

    def loadImages(self, pathNames):
        im_out = []
        for p in pathNames:
            im = sitk.GetArrayFromImage(sitk.ReadImage(p))
            im_out.append(im)

        return im_out

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

    def getAugImagesPath(self, img_nr, aug_nr, z):
        aug_path = self.getAugPath()
        x_folder = '{}input/{}/p{}/'.format(aug_path, self.s.PRE_OR_POST_NAME, img_nr)

        if self.s.GROUND_TRUTH == 'scar_fibrosis':
            y_folder = '{}annotations/'.format(aug_path)
        elif self.s.GROUND_TRUTH == 'left_atrium':
            y_folder = '{}input/{}/p{}/'.format(aug_path, self.s.PRE_OR_POST_NAME, img_nr)

        if not os.path.exists(x_folder):
            os.makedirs(x_folder)
        if not os.path.exists(y_folder):
            os.makedirs(y_folder)

        x_path = '{}de_{}_{}_{}_{}.nii.gz'.format(x_folder, self.s.PRE_OR_POST_XX, img_nr, z, aug_nr)

        if self.s.GROUND_TRUTH == 'scar_fibrosis':
            y_path = '{}staple_{}_{}_{}_{}.nii.gz'.format(y_folder, self.s.PRE_OR_POST_XX, img_nr, z, aug_nr)
        elif self.s.GROUND_TRUTH == 'left_atrium':
            y_path = '{}la_seg_{}_{}_{}_{}.nii.gz'.format(y_folder, self.s.PRE_OR_POST_XX, img_nr, z, aug_nr)

        return x_path, y_path

    def getBOPath(self, model_name):
        return "{}bo.p".format(self.getModelResultsPath(model_name))

    def getNrStepsPath(self, model_name):
        return "{}nr_steps.p".format(self.getModelResultsPath(model_name))

    def getClassWeightAuto(self, y_patches):
        n_pos = np.sum(y_patches)
        n_neg = np.sum((y_patches == 0).astype(int))

        return n_neg / n_pos

    def custom_loss(self, y_true, y_pred):
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
