from settings import Settings
from helper_functions import Helper
from imshow_3D import imshow3D
import SimpleITK as sitk
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import copy
from online_augment import OnlineAugmenter
from scipy import signal
from joblib import Parallel, delayed
import multiprocessing
import cv2


def do_one_iteration(i):
    self = i[0]
    art_nr = i[1]

    no_scar_paths, la_seg_paths, sf_seg_paths = self.h.getNoScarPaths(self.s.NO_SCAR_NRS)
    no_scar_list = self.h.loadImages(no_scar_paths)
    la_seg_list = self.h.loadImages(la_seg_paths)
    sf_seg_list = self.h.loadImages(sf_seg_paths)

    image_spacing = self.h.loadImageSpacing(no_scar_paths)
    print(image_spacing)

    print(no_scar_paths)

    self.s.DEMO = False

    for i in [1]:  # range(len(la_seg_list)):
        self.h.set_image_spacing_xy(image_spacing[i])

        no_scar_full = no_scar_list[i]
        la_seg_full = la_seg_list[i]
        sf_seg_full = sf_seg_list[i]

        art_scar_full = np.zeros(no_scar_full.shape)
        ann_full = np.zeros(no_scar_full.shape)

        for j in range(no_scar_full.shape[0]):

            print('i, art_nr, j = {}, {}, {}'.format(i, art_nr, j))
            no_scar = no_scar_full[j]
            la_seg = la_seg_full[j]
            sf_seg = sf_seg_full[j]

            la_seg = self.pre_process_seg(la_seg)

            scar_removed, sf_seg_dilated = self.remove_scar(no_scar, sf_seg, la_seg)
            scar_removed = self.post_process_art_scar(scar_removed, sf_seg_dilated)

            art_scar_full[j] = scar_removed
            art_scar_full[j], ann_full[j] = self.add_scar(scar_removed, la_seg)

            art_scar_full[j] = self.post_process_art_scar(art_scar_full[j], ann_full[j])

        art_scar_aug, ann_aug, la_seg_aug = OnlineAugmenter(self.s, self.h).augment(
            art_scar_full, ann_full, False, la_seg_full)

        imshow3D(
            np.concatenate(
                (np.concatenate(
                    (no_scar_full, sf_seg_full * np.max(art_scar_aug)
                     ), axis=2
                ),
                np.concatenate(
                    (art_scar_aug, ann_aug * np.max(art_scar_aug)
                     ), axis=2
                )), axis=1
            )
        )

        for j in range(no_scar_full.shape[0]):
            if np.sum(ann_aug[j]) == 0:
                continue

            art_scar_path, ann_path, la_path = self.h.getArtImagesPath(i, art_nr, j, True)

            sitk.WriteImage(sitk.GetImageFromArray(art_scar_aug[j]), art_scar_path)
            sitk.WriteImage(sitk.GetImageFromArray(ann_aug[j]), ann_path)
            sitk.WriteImage(sitk.GetImageFromArray(la_seg_aug[j]), la_path)

class ScarApplier:
    def __init__(self, s, h):
        self.h = h
        self.s = s

    def get_wall(self, bw):
        w = random.randrange(
            int(round(self.h.mm_to_px(self.s.WALL_THICKNESS_MIN_MM))),
            int(round(self.h.mm_to_px(self.s.WALL_THICKNESS_MAX_MM))) + 1
        )

        return bw - sitk.GetArrayFromImage(
            sitk.BinaryErode(
                sitk.GetImageFromArray(bw),
                w
            )
        )

    def get_centroid(self, bw):
        coords = np.argwhere(bw == 1)
        return np.round(np.mean(coords, axis=0))

    def get_random_group(self, centroid, wall):
        angle_a = random.randint(-180, 180)
        angle_b = random.randint(self.s.ANGLE_MIN, self.s.ANGLE_MAX)

        # print('centroid == {}'.format(centroid))
        # print('angle_a == {}'.format(angle_a))
        # print('angle_a + angle_b == {}'.format(angle_a + angle_b))

        group = np.zeros(wall.shape)

        for coord in np.argwhere(wall):
            angle_c = math.atan2(coord[1] - centroid[1], coord[0] - centroid[0]) / (2 * math.pi) * 360
            if angle_a <= angle_c <= angle_a + angle_b\
                    or (angle_a + angle_b) > 180 and angle_c < -180 + (angle_a + angle_b - 180):
                group[coord[0], coord[1]] = 1

        return group

    def pre_process_seg(self, bw):
        bw = sitk.GetArrayFromImage(
            sitk.BinaryMorphologicalClosing(
                sitk.BinaryMorphologicalOpening(
                    sitk.GetImageFromArray(bw), 2
                ), 2
            )
        )
        return bw

    # def blur(self, image):
    #     return sitk.GetArrayFromImage(
    #         sitk.DiscreteGaussian(
    #             sitk.GetImageFromArray(image),
    #             self.s.BLUR_VAR
    #         )
    #     )

    def get_gaussian(self, mri, coord, scale):
        gaussian_val = 0
        w_sum = 0

        kernel_radius = math.ceil(3 * scale)

        for y in range(-kernel_radius, kernel_radius+1):
            for x in range(-kernel_radius, kernel_radius+1):
                w = 1 / (2 * math.pi * scale ** 2) * math.exp(-(x ** 2 + y ** 2)/(2 * scale ** 2))
                w_sum += w
                gaussian_val += w * mri[coord[0] + x, coord[1] + y]

        return gaussian_val

    def blur_local(self, mri, ann):
        dist = sitk.GetArrayFromImage(
            sitk.SignedDanielssonDistanceMap(
                sitk.GetImageFromArray(
                    ann.astype(int)
                )
            )
        )

        max_dist_blur = 5
        max_scale = self.h.mm_to_px(self.s.MAX_SCALE_EDGE_MM)
        scale_cut_off = .7

        scale = copy.copy(dist)
        scale[dist < 0] = 0
        scale = max_scale - scale * max_scale / max_dist_blur
        scale[dist > max_dist_blur] = 0

        scale[scale < scale_cut_off] = 0
        mri_old = copy.copy(mri)

        for c in np.argwhere(scale > 0):
            sc = scale[c[0], c[1]]
            mri[c[0], c[1]] = self.get_gaussian(mri_old, c, sc)

        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.imshow(
        #     np.concatenate((mri_old, mri, ann * np.max(mri_old)), axis=1),
        #     cmap='Greys_r')
        # plt.subplot(2, 1, 2)
        # plt.imshow(scale, cmap='Greys_r')
        # plt.show()

    def blend_in(self, mri, ann):
        dist = sitk.GetArrayFromImage(
            sitk.SignedDanielssonDistanceMap(
                sitk.GetImageFromArray(
                    ann.astype(int)
                )
            )
        )

        impaint_locs = np.ones(dist.shape)
        impaint_locs[dist <= 0] = 0
        impaint_locs[dist >= 2] = 0

        self.h.imshow_demo(impaint_locs)

        blended_in = cv2.inpaint(mri.astype(np.uint16), impaint_locs.astype(np.uint8), 2, cv2.INPAINT_TELEA)

        return blended_in

    def sharpen(self, image):
        return sitk.GetArrayFromImage(
            sitk.LaplacianSharpening(
                sitk.GetImageFromArray(image)
            )
        )

    def get_bp_info(self, mri, la_seg):
        mult = mri * la_seg
        all_bp_values = mult[np.nonzero(mult)]

        bp_mean = np.mean(all_bp_values)
        bp_std = np.std(all_bp_values)

        return bp_mean, bp_std

    def dilate(self, bw, radius):
        return sitk.GetArrayFromImage(
            sitk.BinaryDilate(
                sitk.GetImageFromArray(bw),
                radius
            )
        )

    def get_resampled_random_noise(self, mean, std, sh, resample_factor):
        y_shape = int(math.ceil(sh[0] / resample_factor))
        x_shape = int(math.ceil(sh[1] / resample_factor))

        resampled_sh = (y_shape, x_shape)
        # print('resampled_sh == {}'.format(resampled_sh))
        r = np.random.normal(mean, std, resampled_sh)
        r = r.repeat(math.ceil(resample_factor), axis=0).repeat(math.ceil(resample_factor), axis=1)

        r = r[:sh[0], :sh[1]]

        r = sitk.GetArrayFromImage(
            sitk.DiscreteGaussian(
                sitk.GetImageFromArray(r, 1)
            )
        )

        # plt.figure()
        # plt.imshow(r)
        # plt.show()

        return r

    # Remove the scar that could be in no_scar
    def remove_scar(self, no_scar, sf_seg, la_seg):
        if np.sum(la_seg) == 0:
            return no_scar, sf_seg

        scar_removed = copy.copy(no_scar)

        bp_mean, bp_std = self.get_bp_info(scar_removed, la_seg)

        # print('bp_mean == {}'.format(bp_mean))
        # print('bp_std == {}'.format(bp_std))
        # print('sf_seg.shape == {}'.format(sf_seg.shape))

        sf_seg_dilated = self.dilate(sf_seg, int(round(self.h.mm_to_px(self.s.SF_REMOVE_DILATION_MM))))

        noise = self.get_resampled_random_noise(bp_mean, bp_std * self.s.BP_STD_FACTOR_STD, sf_seg.shape, 2)
        # self.h.mm_to_px(self.s.NOISE_RESAMPLE_FACTOR_MM)

        # print('self.h.mm_to_px(self.s.NOISE_RESAMPLE_FACTOR_MM) == {}'.format(
        # self.h.mm_to_px(self.s.NOISE_RESAMPLE_FACTOR_MM)))
        # print('sf_seg_dilated.shape == {}'.format(sf_seg_dilated.shape))
        # print('noise.shape == {}'.format(noise.shape))

        sf = sf_seg_dilated * noise

        scar_removed[np.nonzero(sf)] = sf[np.nonzero(sf)]

        return scar_removed, sf_seg_dilated

    def post_process_art_scar(self, art_scar, ann):
        art_scar = self.blend_in(art_scar, ann)
        return art_scar

    def add_scar(self, no_scar, la_seg):
        art_scar = no_scar
        ann = np.zeros(no_scar.shape)

        if np.sum(la_seg) == 0:
            return art_scar, ann

        wall = self.get_wall(la_seg)

        centroid = self.get_centroid(la_seg)

        sum = 0
        nb_groups = 0
        for k in range(len(self.s.NB_GROUPS_ODDS)):
            sum += self.s.NB_GROUPS_ODDS[k]
            if random.random() <= sum:
                nb_groups = k
                break

        groups = np.zeros(la_seg.shape)
        for i in range(nb_groups):
            group = self.get_random_group(centroid, wall)
            groups += group

        groups = (groups > 0).astype(int)

        bp_mean, bp_std = self.get_bp_info(no_scar, la_seg)

        sf = groups * self.get_resampled_random_noise(bp_mean + self.s.BP_STD_FACTOR_MEAN * bp_std,
                                                      bp_std * self.s.BP_STD_FACTOR_STD, groups.shape, 2)

        art_scar[np.nonzero(sf)] = sf[np.nonzero(sf)]

        ann[np.nonzero(sf)] += 1

        return art_scar, ann

    def apply(self):
        num_cores = min(10, multiprocessing.cpu_count())
        print('num_cores == {}'.format(num_cores))

        input = [[self, art_nr] for art_nr in range(self.s.NR_ART)]
        do_one_iteration(input[0])
        # Parallel(n_jobs=num_cores)(delayed(do_one_iteration)(i) for i in input)


if __name__ == '__main__':
    s = Settings()
    h = Helper(s)
    ScarApplier(s, h).apply()
