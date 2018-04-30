from settings import *
import random
import numpy as np
import SimpleITK as sitk
import math


# Thanks to http://simpleitk-prototype.readthedocs.io/en/latest/user_guide/transforms/plot_transforms.html
def resample(image, transform):
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 0.0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


# I is gray scale, J should be a binary mask
def augment(I, J):
    if I.shape != J.shape:
        raise Exception('Shape of I does not equal shape of J ({} != {})'.format(I.shape, J.shape))

    rot = random.uniform(ROT_MIN, ROT_MAX) / 360 * 2 * math.pi
    zoom_x = random.uniform(ZOOM_X_MIN, ZOOM_X_MAX)
    zoom_y = random.uniform(ZOOM_Y_MIN, ZOOM_Y_MAX)
    shear_x = random.uniform(SHEAR_X_MIN, SHEAR_X_MAX)
    shear_y = random.uniform(SHEAR_Y_MIN, SHEAR_Y_MAX)
    noise_mean = random.uniform(NOISE_MEAN_MIN, NOISE_MEAN_MAX)
    noise_std = random.uniform(NOISE_STD_MIN, NOISE_STD_MAX)

    affine = sitk.AffineTransform(2)
    affine.Rotate(axis1=0, axis2=1, angle=rot)
    affine.Scale((zoom_x, zoom_y))

    matrix = np.eye(2)
    matrix[0, 1] = -shear_x
    matrix[1, 0] = -shear_y
    affine.SetMatrix(matrix.ravel())

    I_aug = np.zeros(I.shape).astype(int)
    J_aug = np.zeros(J.shape).astype(int)
    for i in range(I.shape[0]):
        I_slice = sitk.GetImageFromArray(I[i])
        J_slice = sitk.GetImageFromArray(I[i])

        I_aug_slice = resample(I_slice, affine)
        J_aug_slice = resample(J_slice, affine)

        I_aug_slice = sitk.AdditiveGaussianNoise(I_aug_slice, noise_mean, noise_std)

        I_aug[i] = I_aug_slice
        J_aug[i] = J_aug_slice

    return I_aug, J_aug
