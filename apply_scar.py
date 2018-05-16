from settings import Settings
from helper_functions import Helper
from imshow_3D import imshow3D
import SimpleITK as sitk
import numpy as np
import random
import matplotlib.pyplot as plt
import math


class ScarApplier:
    def __init__(self, s, h):
        self.h = h
        self.s = s

    def get_wall(self, bw):
        return sitk.GetArrayFromImage(
            sitk.BinaryDilate(
                sitk.GetImageFromArray(bw),
                self.s.WALL_THICKNESS
            )
        ) - bw

    def get_centroid(self, bw):
        coords = np.argwhere(bw == 1)
        return np.round(np.mean(coords, axis=0))

    def get_random_group(self, centroid, wall):
        angle_a = random.randint(-180, 180)
        angle_b = random.randint(self.s.ANGLE_MIN, self.s.ANGLE_MAX)

        print('centroid == {}'.format(centroid))
        print('angle_a == {}'.format(angle_a))
        print('angle_a + angle_b == {}'.format(angle_a + angle_b))

        group = np.zeros(wall.shape)

        for coord in np.argwhere(wall):
            angle_c = math.atan2(coord[1] - centroid[1], coord[0] - centroid[0]) / (2 * math.pi) * 360
            if angle_a <= angle_c <= angle_a + angle_b\
                    or (angle_a + angle_b) > 180 and angle_c < -180 + (angle_a + angle_b - 180):
                group[coord[0], coord[1]] = 1

        return group

    def apply(self):
        no_scar_paths, la_seg_paths = self.h.getNoScarPaths(self.s.NO_SCAR_NRS)
        no_scar_list = self.h.loadImages(no_scar_paths)
        la_seg_list = self.h.loadImages(la_seg_paths)
        # imshow3D(la_seg[0] * no_scar[0])

        for i in range(len(la_seg_list)):
            no_scar_full = no_scar_list[i]
            la_seg_full = la_seg_list[i]

            art_scar_full = np.zeros(no_scar_full.shape)

            for j in range(no_scar_full.shape[0]):
                no_scar = no_scar_full[j]
                la_seg = la_seg_full[j]

                if np.sum(la_seg) == 0:
                    continue

                wall = self.get_wall(la_seg)
                art_scar_full[j] = wall
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

                plt.figure()
                plt.imshow(group + wall, cmap='Greys_r')
                plt.plot(centroid[1], centroid[0], 'r*')
                plt.show()

            imshow3D(art_scar_full)


if __name__ == '__main__':
    s = Settings()
    h = Helper(s)
    ScarApplier(s, h).apply()
