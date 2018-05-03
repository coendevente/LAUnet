from settings import *
from augment import augment
from helper_functions import *
import time
from joblib import Parallel, delayed
import multiprocessing
import random


def offline_augment(img_nr, slices):
    x_aug_path = []
    y_aug_path = []

    for z in slices:
        x_aug_path_z, y_aug_path_z = getAugImagesPath(img_nr, random.randint(0, NR_AUG - 1), z)
        x_aug_path.append(x_aug_path_z)
        y_aug_path.append(y_aug_path_z)

    x_aug_path = np.array(x_aug_path)
    y_aug_path = np.array(y_aug_path)

    x_aug = loadImages([x_aug_path])[0]
    y_aug = loadImages([y_aug_path])[0]

    return x_aug, y_aug


def doOneAug(input):
    i = input[0]
    j = input[1]
    x = input[2]
    y = input[3]
    t0 = input[4]
    l = input[5]

    x_aug, y_aug = augment(x, y, False)

    for z in range(x_aug.shape[0]):
        x_aug_path, y_aug_path = getAugImagesPath(i + 1, j, z)

        sitk.WriteImage(sitk.GetImageFromArray(x_aug[z]), x_aug_path)
        sitk.WriteImage(sitk.GetImageFromArray(y_aug[z]), y_aug_path)

    t_passed = round(time.time() - t0)
    steps_todo = l * NR_AUG
    steps_done = i * NR_AUG + (j + 1)
    ETA = round(t_passed * (steps_todo / steps_done - 1))
    print("{}s passed. ETA is {}. i, j = {}, {}".format(t_passed, ETA, i, j))


def main():
    x_all_path, y_all_path = getImagePaths(range(1, 31))

    x_full_all = loadImages(x_all_path)
    y_full_all = loadImages(y_all_path)

    t0 = time.time()
    inputs = []
    for i in range(len(x_full_all)):
        for j in range(NR_AUG):
            inputs.append([i, j, x_full_all[i], y_full_all[i], t0, len(x_full_all)])

    num_cores = min(8, multiprocessing.cpu_count())
    print('num_cores == {}'.format(num_cores))
    Parallel(n_jobs=num_cores)(delayed(doOneAug)(i) for i in inputs)


if __name__ == "__main__":
    main()