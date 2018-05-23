from settings import *
import random
import numpy as np
import SimpleITK as sitk
import math
from helper_functions import *
from imshow_3D import imshow3D


class OnlineAugmenter():
    def __init__(self, s, h):
        self.s = s  # settings
        self.h = h  # helper functions

    # Thanks to http://simpleitk-prototype.readthedocs.io/en/latest/user_guide/transforms/plot_transforms.html
    def resample(self, image, transform):
        reference_image = image
        interpolator = sitk.sitkLinear
        default_value = 0.0
        return sitk.Resample(image, reference_image, transform,
                             interpolator, default_value)

    def getAffineMatrix(self, rot, zoom_x, zoom_y, shear_x, shear_y):
        matrix = np.eye(2)
        matrix[0, 1] = -shear_x
        matrix[1, 0] = -shear_y

        rotation_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        matrix = np.dot(matrix, rotation_matrix)
        scale_matrix = np.eye(2)
        scale_matrix[0, 0] = zoom_x
        scale_matrix[1, 1] = zoom_y
        matrix = np.dot(matrix, scale_matrix)
        return matrix

    # I is gray scale, J should be a binary mask
    def augment(self, I, J, mx, K):
        if I.shape != J.shape:
            raise Exception('Shape of I does not equal shape of J ({} != {})'.format(I.shape, J.shape))

        rot = random.uniform(self.s.ROT_MIN, self.s.ROT_MAX) / 360 * 2 * np.pi
        zoom_x = random.uniform(self.s.ZOOM_X_MIN, self.s.ZOOM_X_MAX)
        zoom_y = random.uniform(self.s.ZOOM_Y_MIN, self.s.ZOOM_Y_MAX)
        shear_x = random.uniform(self.s.SHEAR_X_MIN, self.s.SHEAR_X_MAX)
        shear_y = random.uniform(self.s.SHEAR_Y_MIN, self.s.SHEAR_Y_MAX)
        noise_mean = random.uniform(self.s.NOISE_MEAN_MIN, self.s.NOISE_MEAN_MAX)
        noise_std = random.uniform(self.s.NOISE_STD_MIN, self.s.NOISE_STD_MAX)
        flip = random.random() < self.s.FLIP_PROB

        if mx:
            rot = self.s.ROT_MAX / 360 * 2 * np.pi
            zoom_x = self.s.ZOOM_X_MIN
            zoom_y = self.s.ZOOM_Y_MIN
            shear_x = self.s.SHEAR_X_MAX
            shear_y = self.s.SHEAR_Y_MAX
            noise_mean = self.s.NOISE_MEAN_MAX
            noise_std = self.s.NOISE_STD_MAX
            flip = True

        if flip:
            I = np.flip(I, axis=2)
            J = np.flip(J, axis=2)

        affine = sitk.AffineTransform(2)

        matrix = self.getAffineMatrix(rot, zoom_x, zoom_y, shear_x, shear_y)
        affine.SetMatrix(matrix.ravel())
        affine.SetCenter((round(I.shape[1] / 2), round(I.shape[2] / 2)))

        I_aug = np.zeros(I.shape).astype(np.uint16)
        J_aug = np.zeros(J.shape).astype(np.uint16)

        if K:
            K_aug = np.zeros(J.shape).astype(np.uint16)
        for i in range(I.shape[0]):
            I_slice = sitk.GetImageFromArray(I[i])
            J_slice = sitk.GetImageFromArray(J[i])

            if K:
                K_slice = sitk.GetImageFromArray(K[i])

            I_aug_slice = self.resample(I_slice, affine)
            J_aug_slice = self.resample(J_slice, affine)

            if K:
                K_aug_slice = self.resample(K_slice, affine)

            I_aug_slice = sitk.AdditiveGaussianNoise(I_aug_slice, noise_mean, noise_std)

            I_aug[i] = sitk.GetArrayFromImage(I_aug_slice)
            J_aug[i] = sitk.GetArrayFromImage(J_aug_slice)

            if K:
                K_aug[i] = sitk.GetArrayFromImage(K_aug_slice)

        # I_aug, J_aug = (I, J)

        if K is not None:
            return I_aug, J_aug, K_aug
        else:
            return I_aug, J_aug

    def test_augment(self):
        x_all_path, y_all_path = self.h.getImagePaths(range(1, 31))
        x_full_all = self.h.loadImages(x_all_path)
        y_full_all = self.h.loadImages(y_all_path)

        nr = 6
        for i in range(5):
            I, J = self.augment(x_full_all[nr], y_full_all[nr], True)
            imshow3D(np.concatenate((x_full_all[nr], I), axis=2))
        # imshow3D(
        #     np.concatenate( (
        #     np.concatenate((x_full_all[nr]/np.max(x_full_all[nr]), y_full_all[nr]), axis=2),
        #         np.concatenate((I / np.max(I), J), axis=2)
        #     ), axis=1)
        # )


if __name__ == "__main__":
    s = Settings()
    h = Helper(s)
    a = OnlineAugmenter(s, h)
    a.test_augment()
