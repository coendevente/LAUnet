from core.augmentations.online_augment import OnlineAugmenter
from core.helper_functions import *
import time
from joblib import Parallel, delayed
import multiprocessing
import random
from core.predict import Predict
import keras
from keras.models import load_model


class OfflineAugmenter:
    def __init__(self, s, h):
        self.s = s
        self.h = h
        self.online_augmenter = OnlineAugmenter(s, h)

    def offline_augment(self, img_nr, slices, get_all, get_lap=True, resize=False):
        x_aug_path = []
        y_aug_path = []
        la_aug_path = []
        lap_aug_path = []

        for z in slices:
            x_aug_path_z, y_aug_path_z, la_aug_path_z, lap_aug_path_z = \
                self.h.getAugImagesPath(img_nr, random.randint(0, self.s.NR_AUG - 1), z, True)

            x_aug_path.append(x_aug_path_z)
            y_aug_path.append(y_aug_path_z)
            la_aug_path.append(la_aug_path_z)
            lap_aug_path.append(lap_aug_path_z)

        x_aug_path = np.array(x_aug_path)
        y_aug_path = np.array(y_aug_path)
        la_aug_path = np.array(la_aug_path)
        lap_aug_path = np.array(lap_aug_path)

        x_aug = self.h.loadImages([x_aug_path])[0]
        y_aug = self.h.loadImages([y_aug_path])[0]
        la_aug = self.h.loadImages([la_aug_path])[0]

        if get_lap:
            lap_aug = self.h.loadImages([lap_aug_path])[0]
        else:
            lap_aug = self.h.loadImages([la_aug_path])[0]

        if resize:
            x_aug = self.h.rescaleImage(x_aug, resize)
            y_aug = self.h.rescaleImage(y_aug, resize)
            la_aug = self.h.rescaleImage(la_aug, resize)
            lap_aug = self.h.rescaleImage(lap_aug, resize)

        if get_all:
            return x_aug, y_aug, la_aug, lap_aug
        elif self.s.GROUND_TRUTH == 'scar_fibrosis':
            return x_aug, y_aug
        elif self.s.GROUND_TRUTH == 'left_atrium':
            return x_aug, la_aug

    def doOneAug(self, input):
        i = input[0]
        j = input[1]
        x = input[2]
        y = input[3]
        la = input[4]
        lap = input[5]
        t0 = input[6]
        l = input[7]

        x_aug, y_aug, la_aug, lap_aug = self.online_augmenter.augment(x, y, False, la, lap)

        for z in range(x_aug.shape[0]):
            x_aug_path, y_aug_path, la_aug_path, lap_aug_path = self.h.getAugImagesPath(i + 1, j, z, True)

            # print(x_aug[z])
            # print(x_aug_path)
            # print("x_aug[z].shape == {}".format(x_aug[z].shape))
            # plt.figure()
            # plt.imshow(x_aug[z])
            # plt.show()
            # print(np.min(x_aug[z]))
            # print(np.unique(x_aug[z]))
            # print('x_aug_path == {}'.format(x_aug_path))
            # print('y_aug_path == {}'.format(y_aug_path))
            # print('la_aug_path == {}'.format(la_aug_path))

            sitk.WriteImage(sitk.GetImageFromArray(x_aug[z]), x_aug_path)
            sitk.WriteImage(sitk.GetImageFromArray(y_aug[z]), y_aug_path)
            sitk.WriteImage(sitk.GetImageFromArray(la_aug[z]), la_aug_path)
            sitk.WriteImage(sitk.GetImageFromArray(lap_aug[z]), lap_aug_path)

        # print(x_aug_path)
        # print(y_aug_path)
        # print(la_aug_path)

        t_passed = round(time.time() - t0)
        steps_todo = l * self.s.NR_AUG
        steps_done = i * self.s.NR_AUG + (j + 1)
        ETA = round(t_passed * (steps_todo / steps_done - 1))
        print("{}s passed. ETA is {}. i, j = {}, {}".format(t_passed, ETA, i, j))

        # imshow3D(x_aug)

    def augment_all(self):
        x_all_path, y_all_path, la_all_path = self.h.getImagePaths(self.s.ALL_NATURAL_SET, True)

        print(x_all_path)
        print(y_all_path)
        print(la_all_path)

        keras.losses.custom_loss = h.custom_loss

        x_full_all = self.h.loadImages(x_all_path)
        y_full_all = self.h.loadImages(y_all_path)
        la_full_all = self.h.loadImages(la_all_path)
        lap_full_all = []

        la_model = load_model(self.h.getModelPath(self.s.MODEL_NAME_FOR_LA_SEG))
        for i in range(len(x_full_all)):
            if self.s.USE_LA_INPUT:
                lap_path = '{}predicted{}.nrrd'.format(self.h.getOfflineAugLAPredictionsPath(self.s.DATA_SET), i)

                print('Loaded {}'.format(lap_path))
                if not self.s.USE_READ_FILE_FOR_LAP:
                    print('Predicting {}'.format(i))
                    x = x_full_all[i]
                    s_la_pred = copy.copy(self.s)
                    s_la_pred.PATCH_SIZE = self.s.MODEL_PS_FOR_LA_SEG
                    s_la_pred.USE_LA_INPUT = False
                    s_la_pred.GROUND_TRUTH = 'left_atrium'
                    s_la_pred.PREDICT_AUX_OUTPUT = False
                    s_la_pred.USE_LA_AUX_LOSS = False
                    s_la_pred.USE_POST_PROCESSING = True
                    prob = Predict(s_la_pred, self.h).predict(x, la_model)
                    prob_thresh = (prob > s.BIN_THRESH).astype(np.uint8)
                    lap = self.h.post_process_la_seg(prob_thresh)

                    sitk.WriteImage(
                        sitk.GetImageFromArray(lap),
                        lap_path
                    )

                    sitk.WriteImage(
                        sitk.GetImageFromArray(x),
                        '{}lge{}.nrrd'.format(self.h.getOfflineAugLAPredictionsPath(self.s.DATA_SET), i)
                    )
                else:
                    lap = sitk.GetArrayFromImage(sitk.ReadImage(lap_path))

                lap_full_all.append(lap)
            else:
                lap_full_all.append(y_full_all[i])

        t0 = time.time()
        inputs = []
        for i in range(len(x_full_all)):
            for j in range(self.s.NR_AUG):
                inputs.append([i, j, x_full_all[i], y_full_all[i], la_full_all[i], lap_full_all[i], t0,
                               len(x_full_all)])

        num_cores = min(8, multiprocessing.cpu_count())
        print('num_cores == {}'.format(num_cores))
        Parallel(n_jobs=num_cores)(delayed(self.doOneAug)(i) for i in inputs)
        # for i in inputs:
        #     self.doOneAug(i)


if __name__ == "__main__":
    s = Settings()
    h = Helper(s)
    a = OfflineAugmenter(s, h)
    a.augment_all()
    s.PRE_OR_POST_XX = 'a'
    s.PRE_OR_POST_NAME = 'pre'
