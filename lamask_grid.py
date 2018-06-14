import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from settings import Settings
from helper_functions import Helper
import copy


def crop_around(im, new_size, center):
    start_idx = (np.array(center) - np.array(new_size) / 2).astype(int)
    out = im[start_idx[0]:start_idx[0]+new_size[0], start_idx[1]:start_idx[1]+new_size[1]]
    return out


def grey2rgb(im):
    return np.stack((im,) * 3, -1)


def get_mask_overlay(im, mask, color, opacity):
    border = mask - sitk.GetArrayFromImage(
        sitk.BinaryErode(
            sitk.GetImageFromArray(mask), 3
        )
    )

    out = copy.copy(im)

    idx = np.argwhere(border == 1)
    out[idx[:, 0], idx[:, 1], :] = ((1 - opacity) * out[idx[:, 0], idx[:, 1], :] + np.array(color) * opacity)

    return out


def normalize(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def get_grid(perc):
    s = Settings()
    h = Helper(s)

    ip = []
    gt = []
    an = []

    nrs = s.VALIDATION_SET

    for nr in nrs:
        path = '{}input_image_{}_-1.nii'.format(h.getModelPredictPath(s.MODEL_NAME), nr)
        im = sitk.GetArrayFromImage(sitk.ReadImage(path))
        ip.append(im)

        path = '{}anno_image_{}_-1.nii'.format(h.getModelPredictPath(s.MODEL_NAME), nr)
        im = sitk.GetArrayFromImage(sitk.ReadImage(path))
        gt.append(im)

        path = '{}prob_thresh_image_{}_-1.nii'.format(h.getModelPredictPath(s.MODEL_NAME), nr)
        im = sitk.GetArrayFromImage(sitk.ReadImage(path))
        an.append(im)

    grid_all = []

    yx_size = (200, 200)

    for i in range(len(nrs)):
        nz = np.argwhere(np.sum(gt[i], axis=(1, 2)) > 0)
        nz = list(np.reshape(nz, nz.shape[:1]))
        print(nz)
        s = nz[int(round(len(nz) * perc))]
        print(s)
        filter = sitk.LabelShapeStatisticsImageFilter()
        filter.Execute(sitk.GetImageFromArray(gt[i][s]))
        center = list(reversed(filter.GetCentroid(1)))

        n = normalize(ip[i][s])
        print(np.min(n))
        print(np.max(n))
        ip_rgb = grey2rgb(n)
        gt_masked = get_mask_overlay(ip_rgb, gt[i][s], [1, 0, 0], 0.5)
        gt_an_masked = get_mask_overlay(gt_masked, an[i][s], [0, 1, 0], 0.5)
        cropped = crop_around(gt_an_masked, yx_size, center)
        grid_all.append(cropped)
        # grid_all.append(crop_around(ip_rgb, yx_size, center))
        # grid_all.append(crop_around(get_mask_overlay(ip_rgb, gt[i][s]), yx_size, center))
        # grid_all.append(crop_around(get_mask_overlay(ip_rgb, an[i][s]), yx_size, center))

        print(grid_all[-1].shape)

    grid_size = [5, 5]

    print(len(grid_all))

    rows = []
    for y in range(grid_size[1]):
        print(y)
        rows.append(np.concatenate(grid_all[y * grid_size[0]:y * grid_size[0] + grid_size[0]], axis=1))
    img_out = np.concatenate(rows, axis=0)

    return img_out


if __name__ == '__main__':
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(get_grid(.25), cmap='Greys_r')

    plt.subplot(1, 3, 2)
    plt.imshow(get_grid(.5), cmap='Greys_r')

    plt.subplot(1, 3, 3)
    plt.imshow(get_grid(.75), cmap='Greys_r')
    plt.show()
