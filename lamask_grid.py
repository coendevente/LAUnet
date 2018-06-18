import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from settings import Settings
from helper_functions import Helper
import copy
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


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

    nrs = s.TESTING_SET
    dice = np.array([0.9006973680422561, 0.910006143806372, 0.9112106116568368, 0.8601871142224925, 0.7259960147934539, 0.7628879251385949, 0.8283370968273197, 0.914378003436467, 0.9261529957159189, 0.8196490038357094, 0.9119802222766654, 0.8743586059357887, 0.8684744959632821, 0.9153005083364386, 0.9223461250934463, 0.8978720592122654, 0.931094379993213, 0.8323133596744007, 0.8952515304300879, 0.7863649833930516, 0.6610801486199576, 0.9001830952523773, 0.905672853299284, 0.8840102160996202, 0.8836265065576483])
    nrs = np.flip(nrs[np.argsort(dice)], 0)  # Sort nrs by Dice score, high to low

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
