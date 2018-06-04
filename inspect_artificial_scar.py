from settings import Settings
from helper_functions import Helper
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from train import Train


class ArtificialScarInspecter:
    def __init__(self, s, h):
        self.s = s
        self.h = h

        self.s.MODEL_NAME = '-'
        self.s.NR_BATCHES = 0
        self.s.ART_FRACTION = 0
        t = Train(s, h)
        t.train()
        t.updateSliceInformation(t.y_full_all, range(1, 31))
        self.t = t

    def get_one(self, artificial):
        if artificial:
            x_path, y_path = self.h.getRandomArtificialPositiveImagePath(False, range(1, 31))

            x = self.h.loadImages([x_path])[0]
            y = self.h.loadImages([y_path])[0]
        else:
            x, y = self.t.getRandomPositivePatch(self.t.x_full_all, self.t.y_full_all, range(1, 31))

            x = self.h.pre_process(x)

            print(x.shape)

            x = np.reshape(x, x.shape[1:])
            y = np.reshape(y, y.shape[1:]).astype(np.uint8)

        print('y.shape == {}'.format(y.shape))

        y = sitk.GetArrayFromImage(
            sitk.BinaryDilate(
                sitk.GetImageFromArray(
                    y
                ), 10
            )
        )

        cc = sitk.ConnectedComponent(sitk.GetImageFromArray(y))
        rand_label = int(np.random.choice(np.unique(sitk.GetArrayFromImage(cc))[1:]))
        # print(rand_label)

        filt = sitk.LabelShapeStatisticsImageFilter()
        filt.Execute(cc)
        c = list(reversed(list(np.round(filt.GetCentroid(rand_label)).astype(np.uint16))))
        print(c)

        pad = 40

        for i in range(2):
            if c[i]-pad < 0:
                strel = [[0, 0], [0, 0]]
                strel[i] = [pad-c[i], 0]
                x = np.pad(x, strel, mode='constant')
                y = np.pad(y, strel, mode='constant')
                c[i] = pad

            if c[i]+pad > x.shape[i]:
                strel = [[0, 0], [0, 0]]
                strel[i] = [0, x.shape[i]-c[i]]
                x = np.pad(x, strel, mode='constant')
                y = np.pad(y, strel, mode='constant')

        x_out = x[c[0]-pad:c[0]+pad, c[1]-pad:c[1]+pad]
        y_out = y[c[0]-pad:c[0]+pad, c[1]-pad:c[1]+pad]

        return self.h.normalize(x_out), y_out

    def get_grid(self, nr_x, nr_y, artificial):
        out1_x = np.array([])
        out1_y = np.array([])
        for i in range(nr_x):
            out2_x = np.array([])
            out2_y = np.array([])
            for j in range(nr_y):
                x, y = self.get_one(artificial)

                out2_x = np.concatenate(
                    (out2_x, x), axis=0
                ) if out2_x.size > 0 else x

                out2_y = np.concatenate(
                    (out2_y, y), axis=0
                ) if out2_y.size > 0 else y

            out1_x = np.concatenate(
                (out1_x, out2_x), axis=1
            ) if out1_x.size > 0 else out2_x

            out1_y = np.concatenate(
                (out1_y, out2_y), axis=1
            ) if out1_y.size > 0 else out2_y

        return out1_x, out1_y

    def inspect(self, nr_x, nr_y):
        natural_grid_x, natural_grid_y = self.get_grid(nr_x, nr_y, False)
        artificial_grid_x, artificial_grid_y = self.get_grid(nr_x, nr_y, True)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(natural_grid_x, cmap='Greys_r')
        plt.contour(natural_grid_y, levels=[.9, 1.1], colors='r', linestyles='-', alpha=.5)
        plt.title('Natural scar')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(artificial_grid_x, cmap='Greys_r')
        plt.contour(artificial_grid_y, levels=[.9, 1.1], colors='r', linestyles='-', alpha=.5)
        plt.title('Artificial scar')

        plt.axis('off')
        plt.show()



if __name__ == '__main__':
    s = Settings()
    h = Helper(s)

    ArtificialScarInspecter(s, h).inspect(4, 4)

    # while True:
    #     try:
    #         ArtificialScarInspecter(s, h).inspect(5, 5)
    #     except ValueError:
    #         print('again')