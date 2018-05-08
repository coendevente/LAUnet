import SimpleITK as sitk
import numpy as np
from settings import *
import os
from keras import backend as K
from imshow_3D import imshow3D
import copy


def getImagePaths(nrs):
    x_all_path = []
    y_all_path = []
    for i in nrs:
        x_all_path.append('{0}input/{1}/p{2}/de_{3}_{2}.nrrd'.format(PATH_TO_DATA, PRE_OR_POST_NAME, i, PRE_OR_POST_XX))
        if GROUND_TRUTH == 'scar_fibrosis':
            y_all_path.append('{0}annotations/staple_{2}_{1}.gipl'.format(PATH_TO_DATA, i, PRE_OR_POST_XX))
        elif GROUND_TRUTH == 'left_atrium':
            y_all_path.append('{0}input/{3}/p{1}/la_seg_{2}_{1}.nrrd'.format(PATH_TO_DATA, i, PRE_OR_POST_XX, PRE_OR_POST_NAME))
    return x_all_path, y_all_path


def loadImages(pathNames):
    im_out = []
    for p in pathNames:
        im = sitk.GetArrayFromImage(sitk.ReadImage(p))
        im_out.append(im)

    return im_out


def cropImage(I, corner, dims):
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


def getModelResultsPath(model_name):
    model_folder = '{}{}/'.format(PATH_TO_MODELS, model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    return model_folder


def getModelPredictPath(model_name):
    predict_folder = '{}predict/'.format(getModelResultsPath(model_name))
    if not os.path.exists(predict_folder):
        os.makedirs(predict_folder)
    return predict_folder


def getModelPath(model_name):
    return '{}model.h5'.format(getModelResultsPath(model_name))


def getModelTrainingTimePath(model_name):
    return '{}training_duration.txt'.format(getModelResultsPath(model_name))


def getModelSettingsPath(model_name):
    return '{}settings.py'.format(getModelResultsPath(model_name))


def getLogPath(model_name):
    return "{}log.p".format(getModelResultsPath(model_name))


def getAugPath():
    path = PATH_TO_AUG
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def getAugImagesPath(img_nr, aug_nr, z):
    aug_path = getAugPath()
    x_folder = '{}input/{}/p{}/'.format(aug_path, PRE_OR_POST_NAME, img_nr)

    if GROUND_TRUTH == 'scar_fibrosis':
        y_folder = '{}annotations/'.format(aug_path)
    elif GROUND_TRUTH == 'left_atrium':
        y_folder = '{}input/{}/p{}/'.format(aug_path, PRE_OR_POST_NAME, img_nr)

    if not os.path.exists(x_folder):
        os.makedirs(x_folder)
    if not os.path.exists(y_folder):
        os.makedirs(y_folder)

    x_path = '{}de_{}_{}_{}_{}.nii.gz'.format(x_folder, PRE_OR_POST_XX, img_nr, z, aug_nr)

    if GROUND_TRUTH == 'scar_fibrosis':
        y_path = '{}staple_{}_{}_{}_{}.nii.gz'.format(y_folder, PRE_OR_POST_XX, img_nr, z, aug_nr)
    elif GROUND_TRUTH == 'left_atrium':
        y_path = '{}la_seg_{}_{}_{}_{}.nii.gz'.format(y_folder, PRE_OR_POST_XX, img_nr, z, aug_nr)

    return x_path, y_path


# def custom_loss(y_true, y_pred):
#     FNentropy = K.binary_crossentropy((y_true - (K.round(y_true) == y_pred)) * y_pred, y_true)
#     FPentropy = K.binary_crossentropy((y_pred - (K.round(y_true) == y_pred)) * y_pred, y_true)
#     TPentropy = K.binary_crossentropy((K.round(y_true) == y_pred) * y_pred, y_true)
#     TNentropy = K.binary_crossentropy((y_pred + y_true - (K.round(y_true) == y_pred)) * y_pred, y_true)
#     return FNentropy * FN_CLASS_WEIGHT + FPentropy + TPentropy + TNentropy


FN_CLASS_WEIGHT = FN_CLASS_WEIGHT_SETTINGS


def set_fn_class_weight(w):
    global FN_CLASS_WEIGHT
    FN_CLASS_WEIGHT = w


def get_fn_class_weight():
    global FN_CLASS_WEIGHT
    return FN_CLASS_WEIGHT


def getClassWeightAuto(y_patches):
    n_pos = np.sum(y_patches)
    n_neg = np.sum((y_patches == 0).astype(int))

    return n_neg / n_pos


def custom_loss(y_true, y_pred):
    y_pred_bw = K.round(y_pred)
    m = y_true - (y_true == y_pred_bw)
    FNentropy = K.binary_crossentropy(m * y_pred, m * y_true)

    m = y_pred - (y_pred_bw == y_pred)
    FPentropy = K.binary_crossentropy(m * y_pred, m * y_true)

    m = y_pred_bw == y_pred
    TPentropy = K.binary_crossentropy(m * y_pred, m * y_true)

    m = y_pred + y_true - (y_pred_bw == y_pred)
    TNentropy = K.binary_crossentropy(m * y_pred, m * y_true)
    print("FN_CLASS_WEIGHT == {}".format(FN_CLASS_WEIGHT))
    return FNentropy * FN_CLASS_WEIGHT + FPentropy + TPentropy + TNentropy
