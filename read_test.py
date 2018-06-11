import SimpleITK as sitk
import time
import numpy as np


if __name__ == '__main__':
    loc_path = '/home/cwdevente/LAUnet/data/input/post/p1/de_b_1.nrrd'
    ext_path = '/data/cwdevente/trash/de_b_1.nrrd'

    n = 100

    loc_times = []
    ext_times = []

    for _ in range(n):
        t_0 = time.time()
        _ = sitk.ReadImage(loc_path)
        loc_times.append(time.time() - t_0)

        t_0 = time.time()
        _ = sitk.ReadImage(ext_path)
        ext_times.append(time.time() - t_0)

    print('loc_time == {} +/- {}'.format(np.mean(loc_times), np.std(loc_times)))
    print('ext_time == {} +/- {}'.format(np.mean(ext_times), np.std(ext_times)))

    print('loc_time / ext_time == {}'.format(np.mean(loc_times) / np.mean(ext_times)))