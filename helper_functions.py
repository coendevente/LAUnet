import SimpleITK as sitk
import numpy as np
from settings import *
import os
from keras import backend as K


def getImagePaths(nrs):
    x_all_path = []
    y_all_path = []
    for i in nrs:
        x_all_path.append('{0}input/{1}/p{2}/de_{3}_{2}.nrrd'.format(PATH_TO_DATA, PRE_OR_POST_NAME, i, PRE_OR_POST_XX))
        y_all_path.append('{0}annotations/staple_{2}_{1}.gipl'.format(PATH_TO_DATA, i, PRE_OR_POST_XX))
    return x_all_path, y_all_path


def loadImages(pathNames):
    im_out = []
    for p in pathNames:
        im = sitk.GetArrayFromImage(sitk.ReadImage(p))
        im_out.append(im)

    return im_out


def cropImage(I, corner, dims):
    d, h, w = dims
    z, y, x = corner
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


def getLossPath(model_name):
    return "{}loss.p".format(getModelResultsPath(model_name))


def custom_loss(y_true, y_pred):
    FNentropy = K.binary_crossentropy((y_true - (K.round(y_true) == y_pred)) * y_pred, y_true)
    FPentropy = K.binary_crossentropy((y_pred - (K.round(y_true) == y_pred)) * y_pred, y_true)
    TPentropy = K.binary_crossentropy((K.round(y_true) == y_pred) * y_pred, y_true)
    TNentropy = K.binary_crossentropy((y_pred + y_true - (K.round(y_true) == y_pred)) * y_pred, y_true)
    return FNentropy * FN_CLASS_WEIGHT + FPentropy + TPentropy + TNentropy