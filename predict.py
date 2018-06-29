from settings import Settings
from helper_functions import Helper
import SimpleITK as sitk
import numpy as np
import keras
from keras.models import load_model
import time
from tkinter import Tk # askopenfilename, asksaveasfilename
from tkinter.filedialog import askopenfilename, asksaveasfilename
import copy
import tensorflow as tf


class Predict:
    def __init__(self, s, h):
        self.s = s
        self.h = h

    def patchCornersFullImage(self, sh):
        step_size = np.subtract(self.s.PATCH_SIZE, self.s.VOXEL_OVERLAP)
        nr_steps = np.divide(sh, step_size)

        # Will be 1 at dimension where they are not rounded numbers, 0 otherwise
        steps_are_not_round = np.array(np.not_equal(nr_steps, np.round(nr_steps) * 1.0), dtype=np.int)
        nr_steps = (np.floor(nr_steps) - steps_are_not_round).astype(int)

        corners_dim = []
        for i in range(3):
            corners_dim.append(np.array(range(nr_steps[i] + 1)) * step_size[i])
            if steps_are_not_round[i]:
                corners_dim[i] = np.append(corners_dim[i], sh[i] - self.s.PATCH_SIZE[i])

            for j in reversed(range(corners_dim[i].shape[0])):
                if corners_dim[i][j] + self.s.PATCH_SIZE[i] > sh[i]:
                    print("i, j == {}, {}".format(i, j))
                    corners_dim[i] = np.delete(corners_dim[i], j)

        patch_corners = []
        for z in corners_dim[0]:
            for y in corners_dim[1]:
                for x in corners_dim[2]:
                    patch_corner = (z, y, x)  # np.multiply((z, y, x), PATCH_SIZE)
                    patch_corners.append(patch_corner)

        return patch_corners

    def patchesFromCorners(self, I, patch_corners):
        patches = []
        for c in patch_corners:
            if self.s.PATCH_SIZE[1] <= I.shape[1]:
                p = self.h.cropImage(I, c, self.s.PATCH_SIZE)
            elif self.s.PATCH_SIZE[1] > I.shape[1]:
                p = self.h.rescaleImage(I[c[0]:c[0] + self.s.PATCH_SIZE[0]], self.s.PATCH_SIZE[1:])

            p = self.h.pre_process(p)

            ps = p.shape
            if self.s.NR_DIM == 2:
                ps = ps[1:]

            p_reshaped = np.reshape(p, (1,) + ps + (1,))

            if self.s.USE_NORMALIZATION:
                p_reshaped = self.h.normalize(p_reshaped)

            patches.append(p_reshaped)
        return patches

    def probPatches(self, patches, model):
        prob_patches = []

        print(len(patches))
        cnt = 0
        for p in patches:
            if cnt % 1 == 0:
                print(cnt)

            # ps = p.shape
            # if self.s.NR_DIM == 2:
            #     ps = ps[1:]
            #
            # if
            #     p_reshaped = np.reshape(p, (1, ) + ps + (1, ))
            #
            # if self.s.USE_NORMALIZATION:
            #     p_reshaped = self.h.normalize(p_reshaped)

            prob_p = model.predict(p)

            if self.s.USE_LA_AUX_LOSS:
                prob_p = prob_p[0] if not self.s.PREDICT_AUX_OUTPUT else prob_p[1]

            prop_p_s = prob_p.shape[1:4]
            if self.s.NR_DIM == 2:
                prop_p_s = (1, ) + prob_p.shape[1:3]
            prob_p_reshaped = np.reshape(prob_p, prop_p_s)
            prob_patches.append(prob_p_reshaped)

            cnt += 1

        return prob_patches

    def fullImageFromPatches(self, sh, prob_patches, patch_corners):
        prob_image = np.zeros(sh)
        count_image = np.zeros(sh)

        for i in range(len(patch_corners)):
            p = prob_patches[i]
            c = list(patch_corners[i])

            for j in range(len(c)):
                if c[j] < 0:
                    c[j] = 0

            # print('c == {}'.format(c))

            if prob_image.shape[1] < p.shape[1]:
                p = self.h.rescaleImage(p, prob_image.shape[1:])

            # print(prob_image.shape)
            # print(p.shape)

            prob_image[c[0]:c[0] + self.s.PATCH_SIZE[0],
                       c[1]:c[1] + self.s.PATCH_SIZE[1],
                       c[2]:c[2] + self.s.PATCH_SIZE[2]] += p

            count_image[c[0]:c[0] + self.s.PATCH_SIZE[0],
                        c[1]:c[1] + self.s.PATCH_SIZE[1],
                        c[2]:c[2] + self.s.PATCH_SIZE[2]] += 1

        # imshow3D(count_image)
        prob_image /= count_image
        return prob_image

    def predict(self, im, model):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        did_rescale = False
        if im.shape[1] < self.s.PATCH_SIZE[1] or self.s.RESIZE_BEFORE_PREDICTION:
            # and self.s.RESIZE_BEFORE_PREDICTION[1] < im.shape[1]):
            print(self.s.RESIZE_BEFORE_PREDICTION)
            did_rescale = True
            old_input_shape = im.shape
            im = self.h.rescaleImage(im, self.s.PATCH_SIZE[1:]) if not self.s.RESIZE_BEFORE_PREDICTION else \
                self.h.rescaleImage(im, self.s.RESIZE_BEFORE_PREDICTION)

        patch_corners = self.patchCornersFullImage(im.shape)
        patches = self.patchesFromCorners(im, patch_corners)

        if self.s.USE_LA_INPUT:
            s_lap = copy.copy(self.s)
            s_lap.USE_LA_INPUT = False
            lap_model = load_model(self.h.getModelPath(self.s.MODEL_NAME_FOR_LA_SEG))
            predict = Predict(s_lap, self.h)
            lap_prob = predict.predict(im, lap_model)

            lap = (lap_prob > self.s.BIN_THRESH).astype(np.uint8)

            if self.s.USE_POST_PROCESSING:
                lap = self.h.post_process_la_seg(lap)

            lap_patches = self.patchesFromCorners(lap, patch_corners)

            for i in range(len(patches)):
                patches[i] = np.concatenate((patches[i], lap_patches[i]), axis=self.s.NR_DIM+1)

        prob_patches = self.probPatches(patches, model)
        prob_image = self.fullImageFromPatches(im.shape, prob_patches, patch_corners)

        if did_rescale:
            print(old_input_shape)
            prob_image = self.h.rescaleImage(prob_image, old_input_shape[1:])
            print(prob_image.shape)

        del sess
        return prob_image


if __name__ == '__main__':
    import easygui

    s = Settings()
    h = Helper(s)
    p = Predict(s, h)

    for i in [1]:  # set(range(18, 26)) - set([11, 17, 23]):
        t0 = time.time()
        keras.losses.custom_loss = h.custom_loss
        model_path = h.getModelPath(s.MODEL_NAME)
        model = load_model(model_path)

        # Tk().withdraw()
        # im_path = askopenfilename(title='Select LGE image')

        # Tk().withdraw()
        # output_file = asksaveasfilename(title='Select output folder')

        # folder = '{}extra/Dataset_case{}/'.format(s.PATH_TO_DATA, i)
        # im_path = '{}lge.nii'.format(folder)

        # Input
        # im_path = '../data/vu_sample/icmr_rhc_pt4.nii.gz'
        # im_path = easygui.fileopenbox()
        # print(im_path)
        im_path = input('Specify input file:')

        # Output
        # output_file = 'la_seg_icmr_rhc_pt4.nii.gz'
        output_file = input('Specify output file (including extension):')


        sim = sitk.ReadImage(im_path)
        im = sitk.GetArrayFromImage(sim)

        h.set_image_spacing_xy(sim.GetSpacing())
        prob = p.predict(im, model)

        # sitk.WriteImage(sitk.GetImageFromArray(prob),
        #                 '{}prob.nii.gz'.format(folder))

        prob_thresh = (prob > s.BIN_THRESH).astype(np.uint8)

        if s.USE_POST_PROCESSING:
            prob_thresh = h.post_process_la_seg(prob_thresh)

        sitk.WriteImage(sitk.GetImageFromArray(prob_thresh), output_file)

        print('Predicting took {} seconds.'.format(time.time() - t0))