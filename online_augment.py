from settings import *
import random
import numpy as np
import SimpleITK as sitk
import math
from helper_functions import *
from imshow_3D import imshow3D
import skimage


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

    def enhance_contrast(self, im, pw):
        # amplitude_pre = np.max(im) - np.min(im)
        # min_pre = np.min(im)
        # im = self.h.normalize(im)
        # im = np.power(im, pw)
        # im = self.h.normalize(im) * amplitude_pre + min_pre

        im = self.h.normalize(im)
        # im = (im.astype(np.float) - np.mean(im)) / (sd_times * np.std(im)) + 1
        im = np.power(im, pw)
        return im


    # I is gray scale, J should be a binary mask
    def augment(self, I, J, mx, K):
        if I.shape != J.shape:
            raise Exception('Shape of I does not equal shape of J ({} != {})'.format(I.shape, J.shape))

        rot = random.uniform(self.s.ROT_MIN, self.s.ROT_MAX) / 360 * 2 * np.pi
        zoom_x = random.uniform(self.s.ZOOM_X_MIN, self.s.ZOOM_X_MAX)
        zoom_y = random.uniform(self.s.ZOOM_Y_MIN, self.s.ZOOM_Y_MAX)
        shear_x = random.uniform(self.s.SHEAR_X_MIN, self.s.SHEAR_X_MAX)
        shear_y = random.uniform(self.s.SHEAR_Y_MIN, self.s.SHEAR_Y_MAX)
        contrast_power = random.uniform(self.s.CONTRAST_POWER_MIN, self.s.CONTRAST_POWER_MAX)
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
            contrast_power = self.s.CONTRAST_POWER_MAX
            flip = False

        if flip:
            I = np.flip(I, axis=2)
            J = np.flip(J, axis=2)

        affine = sitk.AffineTransform(2)

        matrix = self.getAffineMatrix(rot, zoom_x, zoom_y, shear_x, shear_y)
        affine.SetMatrix(matrix.ravel())
        affine.SetCenter((round(I.shape[1] / 2), round(I.shape[2] / 2)))

        I_aug = np.zeros(I.shape).astype(np.float32)
        J_aug = np.zeros(J.shape).astype(np.float32)

        if isinstance(K, np.ndarray):
            K_aug = np.zeros(J.shape).astype(np.uint16)
        for i in range(I.shape[0]):
            I_slice = sitk.GetImageFromArray(I[i])
            J_slice = sitk.GetImageFromArray(J[i])

            if isinstance(K, np.ndarray):
                K_slice = sitk.GetImageFromArray(K[i])

            I_aug_slice = self.resample(I_slice, affine)
            J_aug_slice = self.resample(J_slice, affine)

            if isinstance(K, np.ndarray):
                K_aug_slice = self.resample(K_slice, affine)

            I_aug_slice = sitk.AdditiveGaussianNoise(I_aug_slice, noise_mean, noise_std)
            # I_aug_slice = self.enhance_contrast(sitk.GetArrayFromImage(I_aug_slice), contrast_power)

            # I_aug[i] = I_aug_slice
            I_aug[i] = sitk.GetArrayFromImage(I_aug_slice)
            J_aug[i] = sitk.GetArrayFromImage(J_aug_slice)

            if isinstance(K, np.ndarray):
                K_aug[i] = sitk.GetArrayFromImage(K_aug_slice)

        # I_aug, J_aug = (I, J)

        if isinstance(K, np.ndarray):
            return I_aug, J_aug, K_aug
        else:
            return I_aug, J_aug

    def test_augment(self):
        x_all_path, y_all_path, la_all_path = self.h.getImagePaths(range(1, 31), True)
        x_full_all = self.h.loadImages(x_all_path)
        y_full_all = self.h.loadImages(y_all_path)
        la_full_all = self.h.loadImages(la_all_path)

        imgs = [[], []]

        for nr in range(30):
            for i in range(1):
                I, J, K = self.augment(x_full_all[nr][20:21], y_full_all[nr][20:21], True, la_full_all[nr][20:21])
                # imshow3D(np.concatenate((x_full_all[nr], I), axis=2))

                im_in = [x_full_all[nr][20:21], I]

                for j in range(2):
                    I = np.reshape(im_in[j], im_in[j].shape[1:])
                    I = self.h.normalize(I) * 1000

                    print(np.min(I.astype(np.int16)))
                    print(np.max(I.astype(np.int16)))

                    im_rs = skimage.transform.resize(I.astype(np.int16), (400, 400))
                    print(im_rs.shape)

                    imgs[j].append(im_rs)

                # imshow3D(
                #     np.concatenate(
                #         (
                #             np.concatenate(
                #                 (self.h.normalize(x_full_all[nr][sr]), y_full_all[nr][sr]), axis=2
                #             ),
                #             np.concatenate(
                #                 (self.h.normalize(I), J), axis=2
                #             )
                #         )
                #         , axis=1
                #     )
                # )

        print(len(imgs[0]))
        print(len(imgs[1]))

        plt.figure()
        plt.subplot(1, 2, 1)
        g = self.h.get_grid(6, 5, imgs[0])
        plt.imshow(g, cmap='Greys_r')

        plt.subplot(1, 2, 2)
        g = self.h.get_grid(6, 5, imgs[1])
        plt.imshow(g, cmap='Greys_r')
        plt.show()


if __name__ == "__main__":
    s = Settings()
    h = Helper(s)
    a = OnlineAugmenter(s, h)
    a.test_augment()
