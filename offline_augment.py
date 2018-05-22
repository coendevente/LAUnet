from settings import *
from online_augment import OnlineAugmenter
from helper_functions import *
import time
from joblib import Parallel, delayed
import multiprocessing
import random
import matplotlib.pyplot as plt


class OfflineAugmenter:
    def __init__(self, s, h):
        self.s = s
        self.h = h
        self.online_augmenter = OnlineAugmenter(s, h)

    def offline_augment(self, img_nr, slices, get_all):
        x_aug_path = []
        y_aug_path = []
        la_aug_path = []

        for z in slices:
            x_aug_path_z, y_aug_path_z, la_aug_path_z = \
                self.h.getAugImagesPath(img_nr, random.randint(0, self.s.NR_AUG - 1), z, True)
            x_aug_path.append(x_aug_path_z)
            y_aug_path.append(y_aug_path_z)
            la_aug_path.append(la_aug_path_z)

        x_aug_path = np.array(x_aug_path)
        y_aug_path = np.array(y_aug_path)
        la_aug_path = np.array(la_aug_path)

        x_aug = self.h.loadImages([x_aug_path])[0]
        y_aug = self.h.loadImages([y_aug_path])[0]
        la_aug = self.h.loadImages([la_aug_path])[0]

        if get_all:
            return x_aug, y_aug, la_aug
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
        t0 = input[5]
        l = input[6]

        x_aug, y_aug, la_aug = self.online_augmenter.augment(x, y, False, la)

        for z in range(x_aug.shape[0]):
            x_aug_path, y_aug_path, la_aug_path = self.h.getAugImagesPath(i + 1, j, z, True)

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

        # print(x_aug_path)
        # print(y_aug_path)
        # print(la_aug_path)

        t_passed = round(time.time() - t0)
        steps_todo = l * self.s.NR_AUG
        steps_done = i * self.s.NR_AUG + (j + 1)
        ETA = round(t_passed * (steps_todo / steps_done - 1))
        print("{}s passed. ETA is {}. i, j = {}, {}".format(t_passed, ETA, i, j))

    def augment_all(self):
        x_all_path, y_all_path, la_all_path = self.h.getImagePaths(self.s.ALL_NATURAL_SET, True)

        print(x_all_path)
        print(y_all_path)
        print(la_all_path)

        x_full_all = self.h.loadImages(x_all_path)
        y_full_all = self.h.loadImages(y_all_path)
        la_full_all = self.h.loadImages(la_all_path)

        t0 = time.time()
        inputs = []
        for i in range(len(x_full_all)):
            for j in range(self.s.NR_AUG):
                inputs.append([i, j, x_full_all[i], y_full_all[i], la_full_all[i], t0, len(x_full_all)])

        num_cores = min(2, multiprocessing.cpu_count())
        print('num_cores == {}'.format(num_cores))
        Parallel(n_jobs=num_cores)(delayed(self.doOneAug)(i) for i in inputs)
        # for i in inputs:
        #     self.doOneAug(i)


if __name__ == "__main__":
    s = Settings()
    h = Helper(s)
    a = OfflineAugmenter(s, h)
    a.augment_all()
