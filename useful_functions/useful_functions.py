# Generally useful functions
#
# Author: Coen de Vente
#
#  --------------------------
# | Most important functions |
#  --------------------------
# vis_mask(x, y, slice_number=False):       visualise a binary image y on gray image x (will show border)
# vis_around_center(x, y, window_size):     visualise a binary image around the center of a 3D image
# np_image(p):                              read an image into numpy array directly from path name p


import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import copy


def get_centroid(binary_mask):
    filter = sitk.LabelShapeStatisticsImageFilter()
    filter.Execute(sitk.GetImageFromArray(binary_mask))
    centroid = list(reversed(filter.GetCentroid(1)))

    return centroid


def get_center_2D(binary_mask):
    coords = np.argwhere(binary_mask == 1)
    all_y, all_x = [c[0] for c in coords], [c[1] for c in coords]
    min_y, min_x, max_y, max_x = min(all_y), min(all_x), max(all_y), max(all_x)
    center_y = int((min_y + max_y) / 2)
    center_x = int((min_x + max_x) / 2)
    return (center_y, center_x)


def crop_around_point(im, new_size, point):
    start_idx = (np.array(point) - np.array(new_size) / 2).astype(int)
    out = im[start_idx[0]:start_idx[0]+new_size[0], start_idx[1]:start_idx[1]+new_size[1]]
    return out


def crop_around_center(im, new_size):
    center = (np.array(im.shape) / 2).astype(np.int)
    return crop_around_point(im, new_size, center)


def normalize(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def get_mask_overlay(im, mask, color=(1, 1, 0), opacity=.5):
    out = copy.copy(im)
    out = normalize(out)
    out = grey2rgb(out)

    idx = np.argwhere(mask == 1)
    out[idx[:, 0], idx[:, 1], :] = ((1 - opacity) * out[idx[:, 0], idx[:, 1], :] + np.array(color) * opacity)

    return out


def grey2rgb(im):
    return np.stack((im,) * 3, -1)


def get_border(y, b=3):
    return y - sitk.GetArrayFromImage(
        sitk.BinaryErode(
            sitk.GetImageFromArray(y),
            b
        )
    )


def vis_mask(x, y, slice_number=False):
    plt.figure()

    if not slice_number:
        xc, yc = x, y
    else:
        xc, yc = x[slice_number], y[slice_number]

    plt.imshow(get_mask_overlay(xc, get_border(yc)))
    plt.show()


def np_image(p):
    return sitk.GetArrayFromImage(sitk.ReadImage(p))


# Pretty useful function, x and y are 3D, visualises middle slice, around centroid of annotation
def vis_around_center(x, y, window_size=(100, 100)):
    slice_number = int(x.shape[0] / 2)
    ys = y[slice_number]
    center = get_center_2D(ys)
    xc = crop_around_point(x[slice_number], window_size, center)
    yc = crop_around_point(ys, window_size, center)

    vis_mask(xc, yc)


if __name__ == '__main__':
    xp = '/Users/coendevente/Desktop/Master/jaar_1/externship/LAUnet/challenge_2018_data/input/3C2QTUNI0852XV7ZH4Q1/' +\
         'lgemri.nrrd'
    yp = '/Users/coendevente/Desktop/Master/jaar_1/externship/LAUnet/challenge_2018_data/input/3C2QTUNI0852XV7ZH4Q1/' +\
         'laendo.nrrd'
    x = np_image(xp)
    y = np_image(yp)

    vis_around_center(x, y, (180, 180))
