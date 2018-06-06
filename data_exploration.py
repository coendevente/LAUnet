import numpy as np
import SimpleITK as sitk
from itertools import chain
import matplotlib.pyplot as plt
from settings import *


class DataExploration:
    def __init__(self, s):
        self.s = s

    def data_explore(self):
        # Nr of ones in the annotations
        nrOfOnes = {'a': [[], []], 'b': [[], []]}
        total_pixels = {'a': 0, 'b': 0}

        # for xx_name in [['a', 'pre'], ['b', 'post']]:
        for xx_name in [['b', 'post']]:
            # Get image sizes
            allShapes = {}
            allYXDim = {}

            for i in range(1, 44):
                if i == 20 and xx_name[0] == 'a' or i == 18 and xx_name[0] == 'b':
                    continue
                    # Sizes annotation and input of i == 20 are not equal

                path_ann = '{0}annotations/kcl_{1}_{2}.nrrd'.format(self.s.PATH_TO_DATA, xx_name[0], i)
                path_input = '{0}input/{2}/p{1}/de_{3}_{1}.nrrd'.format(self.s.PATH_TO_DATA, i, xx_name[1], xx_name[0])

                I_ann = sitk.GetArrayFromImage(sitk.ReadImage(path_ann))
                I_input = sitk.GetArrayFromImage(sitk.ReadImage(path_input))

                nrOfOnes[xx_name[0]][0].append(i)
                nrOfOnes[xx_name[0]][1].append(np.sum(I_ann))

                sh_ann = I_ann.shape
                sh_input = I_input.shape

                if sh_ann != sh_input:
                    raise Exception('size of {} != size of {}, ({} != {})'.format(path_ann, path_input, sh_ann,
                                                                                  sh_input))

                if sh_ann in allShapes:
                    allShapes[sh_ann][0] += 1
                    allShapes[sh_ann][1].append(path_input)
                else:
                    allShapes[sh_ann] = [1, [path_input]]

                if sh_ann[1:] in allYXDim:
                    allYXDim[sh_ann[1:]][0] += 1
                    allYXDim[sh_ann[1:]][1].append(path_input)
                else:
                    allYXDim[sh_ann[1:]] = [1, [path_input]]

                total_pixels[xx_name[0]] += np.prod(I_input.shape)

            print("Unique shapes in {} ({}): {}".format(xx_name[1], len(allShapes), allShapes))
            print("Unique YX dimensions in {} ({}): {}".format(xx_name[1], len(allYXDim), allYXDim))

            for yx in allYXDim:
                print(yx)
                toShow = []
                for path_input in allYXDim[yx][1]:
                    I_input = sitk.GetArrayFromImage(sitk.ReadImage(path_input))
                    I_input = I_input[round(I_input.shape[0] / 2)]
                    I_input = I_input / np.max(I_input)

                    toShow.append(I_input)

                # I_out = sitk.GetImageFromArray(np.array(np.concatenate(toShow, axis=1) * 255, dtype=np.uint8))
                # sitk.WriteImage(I_out, "{0}data_exploration/{1}_shape={2}.png".format(PATH_TO_RESULTS, xx_name[1], yx))

        # print("Pre: Minimum nr of positive voxels: {} has {}".format(np.argmin(nrOfOnes['a'])+1, np.min(nrOfOnes['a'])))
        # print("Pre: Maximum nr of positive voxels: {} has {}".format(np.argmax(nrOfOnes['a'])+1, np.max(nrOfOnes['a'])))

        print("Post: Minimum nr of positive voxels: {} has {}".format(np.argmin(nrOfOnes['b'])+1, np.min(nrOfOnes['b'])))
        print("Post: Maximum nr of positive voxels: {} has {}".format(np.argmax(nrOfOnes['b'])+1, np.max(nrOfOnes['b'])))

        # print("Pre: Total number of foreground pixels: {}".format(np.sum(nrOfOnes['a'])))
        # print("Pre: Total number of pixels: {}".format(total_pixels['a']))

        print("Post: Total number of foreground pixels: {}".format(np.sum(nrOfOnes['b'])))
        print("Post: Total number of pixels: {}".format(total_pixels['b']))

        nrOfOnesNumpy = np.array(nrOfOnes['b'][1])
        print(nrOfOnesNumpy)
        nrOfOnesSorted = np.sort(nrOfOnesNumpy)
        nrOfOnesArgSorted = [nrOfOnes['b'][0][i] for i in np.argsort(nrOfOnesNumpy)]

        plt.figure()
        plt.scatter(range(len(nrOfOnesSorted)), nrOfOnesSorted)

        training = []
        validation = []
        testing = []
        train_per_step = [3, 3, 3, 3, 3]
        val_per_step = [1, 2, 1, 2, 1]
        print(nrOfOnesArgSorted)
        for i in range(0, 5):
            p = np.random.permutation(nrOfOnesArgSorted[i * 6: (i+1)*6])
            training += list(p[0:train_per_step[i]])
            validation += list(p[train_per_step[i]:train_per_step[i]+val_per_step[i]])
            testing += list(p[train_per_step[i]+val_per_step[i]:])

        print(training)
        print(validation)
        print(testing)

        for i, txt in enumerate(nrOfOnesArgSorted):
            plt.annotate(txt, (i, nrOfOnesSorted[i]))
        plt.show()

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.hist(nrOfOnes['a'], bins=40)
        plt.subplot(2, 1, 2)
        plt.hist(nrOfOnes['b'], bins=40)
        plt.show()


if __name__ == "__main__":
    s = Settings()
    DataExploration(s).data_explore()
