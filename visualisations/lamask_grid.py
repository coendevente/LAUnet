import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from core.settings import Settings
from core.helper_functions import Helper
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


def get_grid():
    s = Settings()
    h = Helper(s)

    ip = []
    gt = []
    an = []

    nrs = np.array([70, 21, 95, 73, 78, 26, 38, 82, 47, 40, 66, 59, 13, 89, 71, 88, 37, 22, 84, 10, 97, 68, 65, 48,
                    45])

    d = np.array([0.9009270902037788, 0.9104197765530493, 0.9128334854875481, 0.8607061285160114, 0.726180976928685,
                  0.7618735476244846, 0.8426088283800738, 0.9227242238885163, 0.9267448462842333, 0.8202146853529186,
                  0.9124323842524247, 0.8758631939535643, 0.8686964143471794, 0.9156216299184503, 0.9226021312080136,
                  0.8982460315886207, 0.9316061013262126, 0.8248859357030646, 0.8955985800466059, 0.7870071142712975,
                  0.6458948916498899, 0.9089561365052262, 0.9061868164772646, 0.8842184960264304, 0.8842468629005924])
    nrs = np.flip(nrs[np.argsort(d)], 0)

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
        # nz = np.argwhere(np.sum(gt[i], axis=(1, 2)) > 0)
        # nz = list(np.reshape(nz, nz.shape[:1]))
        # print(nz)
        # s = nz[int(round(len(nz) * perc))]
        # print(s)
        s = 44
        filter = sitk.LabelShapeStatisticsImageFilter()
        filter.Execute(sitk.GetImageFromArray(gt[i][s]))
        center = list(reversed(filter.GetCentroid(1)))

        # print(np.min(n))
        # print(np.max(n))
        ip_rgb = grey2rgb(ip[i][s])
        cropped = crop_around(ip_rgb, yx_size, center)
        cropped = normalize(cropped)
        gt_cropped = crop_around(gt[i][s], yx_size, center)
        an_cropped = crop_around(an[i][s], yx_size, center)
        gt_masked = get_mask_overlay(cropped, gt_cropped, [1, 0, 0], 0.5)
        gt_an_masked = get_mask_overlay(gt_masked, an_cropped, [0, 1, 0], 0.5)

        gt_an_masked = np.pad(gt_an_masked, ((10, 10), (10, 10), (0, 0)), mode='constant', constant_values=255)
        grid_all.append(gt_an_masked)
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
    # plt.subplot(1, 3, 1)
    # plt.imshow(get_grid(.25), cmap='Greys_r')

    # plt.subplot(1, 3, 2)
    plt.imshow(get_grid(), cmap='Greys_r')

    # plt.subplot(1, 3, 3)
    # plt.imshow(get_grid(.75), cmap='Greys_r')
    plt.axis('off')
    plt.show()
