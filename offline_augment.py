from settings import *
from online_augment import OnlineAugmenter
from helper_functions import *
import time
from joblib import Parallel, delayed
import multiprocessing
import random
import matplotlib.pyplot as plt


class OfflineAugmenter():
    def __init__(self, s, h):
        self.s = s
        self.h = h
        self.online_augmenter = OnlineAugmenter(s, h)

    def offline_augment(self, img_nr, slices):
        x_aug_path = []
        y_aug_path = []

        for z in slices:
            x_aug_path_z, y_aug_path_z = self.h.getAugImagesPath(img_nr, random.randint(0, self.s.NR_AUG - 1), z)
            x_aug_path.append(x_aug_path_z)
            y_aug_path.append(y_aug_path_z)

        x_aug_path = np.array(x_aug_path)
        y_aug_path = np.array(y_aug_path)

        x_aug = self.h.loadImages([x_aug_path])[0]
        y_aug = self.h.loadImages([y_aug_path])[0]

        return x_aug, y_aug

    def doOneAug(self, input):
        i = input[0]
        j = input[1]
        x = input[2]
        y = input[3]
        t0 = input[4]
        l = input[5]

        x_aug, y_aug = self.online_augmenter.augment(x, y, False)

        for z in range(x_aug.shape[0]):
            x_aug_path, y_aug_path = self.h.getAugImagesPath(i + 1, j, z)

            # print(x_aug[z])
            # print(x_aug_path)
            # print("x_aug[z].shape == {}".format(x_aug[z].shape))
            # plt.figure()
            # plt.imshow(x_aug[z])
            # plt.show()
            # print(np.min(x_aug[z]))
            # print(np.unique(x_aug[z]))
            sitk.WriteImage(sitk.GetImageFromArray(x_aug[z]), x_aug_path)
            sitk.WriteImage(sitk.GetImageFromArray(y_aug[z]), y_aug_path)

        t_passed = round(time.time() - t0)
        steps_todo = l * self.s.NR_AUG
        steps_done = i * self.s.NR_AUG + (j + 1)
        ETA = round(t_passed * (steps_todo / steps_done - 1))
        print("{}s passed. ETA is {}. i, j = {}, {}".format(t_passed, ETA, i, j))

    def augment_all(self):
        x_all_path, y_all_path = self.h.getImagePaths(range(1, 31))

        x_full_all = self.h.loadImages(x_all_path)
        y_full_all = self.h.loadImages(y_all_path)

        t0 = time.time()
        inputs = []
        for i in range(len(x_full_all)):
            for j in range(self.s.NR_AUG):
                inputs.append([i, j, x_full_all[i], y_full_all[i], t0, len(x_full_all)])

        num_cores = min(8, multiprocessing.cpu_count())
        print('num_cores == {}'.format(num_cores))
        Parallel(n_jobs=num_cores)(delayed(self.doOneAug)(i) for i in inputs)

if __name__ == "__main__":
    s = Settings()
    h = Helper(s)
    a = OfflineAugmenter(s, h)
    a.augment_all()
