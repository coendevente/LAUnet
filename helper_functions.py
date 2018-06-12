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
from skimage import io
from skimage.transform import resize


class ArtificialPaths:
    paths = {'a': {}, 'b': {}}

    def __init__(self, s, h):
        self.s = s
        self.h = h

    def get_image_slices(self, xx, img_nr):
        aug_path = self.h.getArtPath()
        x_folder = '{}input/'.format(aug_path)

        if img_nr not in self.paths[xx]:
            self.paths[xx][img_nr] = glob.glob('{}de_{}_{}_*.nii.gz'.format(x_folder, xx, img_nr))

            for i in range(len(self.paths[xx][img_nr])):
                self.paths[xx][img_nr][i] = self.paths[xx][img_nr][i].replace('\\', '/')

        return self.paths[xx][img_nr]


class Helper():
    def __init__(self, s):
        self.s = s
        self.image_spacing_xy = -1

        self.artificial_paths = ArtificialPaths(s, self)

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

    def get_image_paths_original(self, nrs, get_all):
        x_all_path = []
        y_all_path = []
        la_all_path = []

        for i in nrs:
            x_all_path.append('{0}input/{1}/p{2}/de_{3}_{2}.nrrd'.format(self.s.PATH_TO_DATA, self.s.PRE_OR_POST_NAME,
                                                                         i, self.s.PRE_OR_POST_XX))

            y_all_path.append('{0}annotations/kcl_{2}_{1}.nrrd'.format(self.s.PATH_TO_DATA, i,
                                                                          self.s.PRE_OR_POST_XX))

            la_all_path.append('{0}input/{3}/p{1}/la_seg_{2}_{1}.nrrd'.format(self.s.PATH_TO_DATA, i,
                                                                             self.s.PRE_OR_POST_XX,
                                                                             self.s.PRE_OR_POST_NAME))

        if get_all:
            return x_all_path, y_all_path, la_all_path
        elif self.s.GROUND_TRUTH == 'scar_fibrosis':
            return x_all_path, y_all_path
        elif self.s.GROUND_TRUTH == 'left_atrium':
            return x_all_path, la_all_path

    def get_image_paths_challenge_2018(self, nrs, get_all):
        if self.s.GROUND_TRUTH == 'scar_fibrosis':
            raise Exception('No scar annotations in challenge 2018 data')

        x_path = []
        la_path = []

        x_all_path = sorted(glob.glob('{}input/*/lgemri.nrrd'.format(self.s.PATH_TO_DATA)))

        for i in np.array(nrs) - 1:
            x_path.append(x_all_path[i])
            la_path.append(x_path[-1].replace('lgemri', 'laendo'))

        if get_all:
            return x_path, la_path, la_path
        else:
            return x_path, la_path

    def getImagePaths(self, nrs, get_all):
        if self.s.DATA_SET == 'original':
            return self.get_image_paths_original(nrs, get_all)
        elif self.s.DATA_SET == 'challenge_2018':
            return self.get_image_paths_challenge_2018(nrs, get_all)

    def loadImages(self, pathNames):
        im_out = []
        for p in pathNames:
            im_sitk = sitk.ReadImage(p)
            im = sitk.GetArrayFromImage(im_sitk)

            self.set_image_spacing_xy(self.loadImageSpacing([p])[0])
            im_out.append(im)

        return im_out

    def loadImageSpacing(self, pathNames):
        spacing = []
        for p in pathNames:
            spacing.append(
                sitk.ReadImage(p).GetSpacing()
            )

        return spacing

    def rescaleImage(self, I, dims):
        I = self.normalize(I)
        I_out = np.zeros(tuple([I.shape[0]]) + dims)
        for i in range(I.shape[0]):
            I_out[i] = resize(I[i], dims)

        return I_out

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
        y_path = '{}kcl_{}_{}_{}_{}.nii.gz'.format(y_folder, self.s.PRE_OR_POST_XX, img_nr, z, aug_nr)
        la_path = '{}la_seg_{}_{}_{}_{}.nii.gz'.format(la_folder, self.s.PRE_OR_POST_XX, img_nr, z, aug_nr)
        lap_path = '{}lap_seg_{}_{}_{}_{}.nii.gz'.format(la_folder, self.s.PRE_OR_POST_XX, img_nr, z, aug_nr)

        if get_all:
            return x_path, y_path, la_path, lap_path
        elif self.s.GROUND_TRUTH == 'left_atrium':
            return x_path, la_path
        elif self.s.GROUND_TRUTH == 'scar_fibrosis':
            return x_path, y_path

    def getRandomArtificialPositiveImagePath(self, get_all, set_idx):
        aug_path = self.getArtPath()

        x_folder = '{}input/'.format(aug_path)
        y_folder = '{}annotations/'.format(aug_path)
        la_folder = '{}input/'.format(aug_path)

        img_nrs_pre = list(self.s.NO_SCAR_NRS_PRE)
        img_nrs_post = list(set(set_idx).intersection(self.s.NO_SCAR_NRS_POST))

        img_nrs = img_nrs_pre + img_nrs_post
        xxs = ['a'] * len(img_nrs_pre) + ['b'] * len(img_nrs_post)

        random_i = random.randint(0, len(img_nrs) - 1)
        # art_nr = random.randint(0, self.s.NR_ART - 1)

        # print('img_nrs == {}'.format(img_nrs))
        # print('xxs == {}'.format(xxs))

        # t0 = time.time()
        # x_path = random.choice(
        #     glob.glob('{}de_{}_*_{}.nii.gz'.format(x_folder, img_nr, art_nr))
        # ).replace('\\', '/')
        # print('loading took {}'.format(time.time() - t0))

        # print('random_i == {}'.format(random_i))
        # print('xxs[random_i] == {}'.format(xxs[random_i]))
        # print('img_nrs[random_i] == {}'.format(img_nrs[random_i]))

        x_path = random.choice(self.artificial_paths.get_image_slices(xxs[random_i], img_nrs[random_i]))

        y_path = x_path.replace(x_folder, y_folder)
        y_path = y_path.replace('de_', 'kcl_')
        la_path = x_path.replace(x_folder, la_folder)
        la_path = la_path.replace('de_', 'la_seg_')

        if get_all:
            return x_path, y_path, la_path
        elif self.s.GROUND_TRUTH == 'left_atrium':
            return x_path, la_path
        elif self.s.GROUND_TRUTH == 'scar_fibrosis':
            return x_path, y_path

    def getArtImagesPath(self, xx, img_nr, art_nr, z, get_all):
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

        x_path = '{}de_{}_{}_{}_{}.nii.gz'.format(x_folder, xx, img_nr, z, art_nr)
        y_path = '{}kcl_{}_{}_{}_{}.nii.gz'.format(y_folder, xx, img_nr, z, art_nr)
        la_path = '{}la_seg_{}_{}_{}_{}.nii.gz'.format(la_folder, xx, img_nr, z, art_nr)

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

    def getNoScarPaths(self, nrs_pre, nrs_post):
        no_scar_paths = []
        la_seg_paths = []
        sf_seg_paths = []
        for i in nrs_pre:
            p_folder = self.s.PATH_TO_NO_SCAR_PRE + 'p{}/'.format(i)
            sf_folder = '{}annotations/'.format(self.s.PATH_TO_DATA)

            no_scar_paths.append(p_folder + 'de_a_{}.nrrd'.format(i))
            la_seg_paths.append(p_folder + 'la_seg_a_{}.nrrd'.format(i))
            sf_seg_paths.append(sf_folder + 'kcl_a_{}.nrrd'.format(i))

        for i in nrs_post:
            p_folder = self.s.PATH_TO_NO_SCAR_POST + 'p{}/'.format(i)
            sf_folder = '{}annotations/'.format(self.s.PATH_TO_DATA)

            no_scar_paths.append(p_folder + 'de_b_{}.nrrd'.format(i))
            la_seg_paths.append(p_folder + 'la_seg_b_{}.nrrd'.format(i))
            sf_seg_paths.append(sf_folder + 'kcl_b_{}.nrrd'.format(i))
        return no_scar_paths, la_seg_paths, sf_seg_paths

    def imshow_demo(self, im):
        if self.s.DEMO:
            plt.figure()
            # plt.imshow(im[241:361, 146:342], cmap='Greys_r')
            plt.imshow(im, cmap='Greys_r')
            plt.show()

    # Thanks to https://github.com/keras-team/keras/issues/3611
    def dice_coef(self, y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1-self.dice_coef(y_true, y_pred)

    def normalize(self, im):
        # return (im - np.mean(im)) / np.std(im)
        # print(im.shape)
        return (im - np.min(im)) / (np.max(im) - np.min(im))

    def normalize_multiple_ndarray(self, ls_in):
        ls_out = np.zeros(ls_in.shape)
        for i in range(ls_in.shape[0]):
            ls_out[i] = self.normalize(ls_in[i])

        return ls_out

    def normalize_multiple_list(self, ls_in):
        ls_out = []
        for i in range(len(ls_in)):
            ls_out.append(self.normalize(ls_in[i]))

        return ls_out

    def pre_process(self, x):
        if not self.s.USE_PRE_PROCESSING:
            return x

        x_pre_processed = sitk.GetArrayFromImage(
            sitk.RecursiveGaussian(
                sitk.GetImageFromArray(x), self.mm_to_px(self.s.BLUR_SCALE_MM)
            )
        )

        return x_pre_processed

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

    def smooth(self, x, w):
        pre = math.floor(w / 2)
        post = w - pre
        for i in range(len(x)):  # range(pre, len(x) - post):
            s = max(0, i - pre)
            e = min(len(x), i + post)
            x[i] = np.mean(x[s:e])
        return x

    def get_grid(self, nr_x, nr_y, imgs):
        out1_x = np.array([])
        for i in range(nr_x):
            out2_x = np.array([])
            for j in range(nr_y):
                x = imgs[i * nr_y + j]

                out2_x = np.concatenate(
                    (out2_x, x), axis=0
                ) if out2_x.size > 0 else x

            out1_x = np.concatenate(
                (out1_x, out2_x), axis=1
            ) if out1_x.size > 0 else out2_x

        return out1_x

    def resize_to_unet_shape(self, im, two_power):
        ys = im.shape[0]
        xs = im.shape[1]
        factor = 2 ** two_power
        ys_new = int(math.ceil(ys / factor) * factor)
        xs_new = int(math.ceil(xs / factor) * factor)

        im = resize(im, (ys_new, xs_new))

        return im

    def post_process_la_seg(self, la):
        sla = sitk.GetImageFromArray(la)
        cc = sitk.GetArrayFromImage(sitk.ConnectedComponent(sla, True))

        labels = list(np.unique(cc)[1:])

        label_count = []
        for l in labels:
            label_count.append(np.sum(cc == l))

        label_most = np.argmax(label_count) + 1

        la_out = (cc == label_most).astype(np.uint8)

        return la_out
