import numpy as np
import SimpleITK as sitk
from itertools import chain
import matplotlib.pyplot as plt

def main():
    # Path to data folder
    path_to_data = '../data/'
    path_to_results = '../results/'

    # Get image sizes
    allShapes = {}
    allYXDim = {}
    for xx_name in [['a', 'pre'], ['b', 'post']]:
        for i in range(1, 31):
            if i == 20 and xx_name[0] == 'a':
                continue
                # Sizes annotation and input of i == 20 are not equal
            path_ann = '{0}annotations/staple_{1}_{2}.gipl'.format(path_to_data, xx_name[0], i)
            I_ann = sitk.GetArrayFromImage(sitk.ReadImage(path_ann))
            sh_ann = I_ann.shape
            print(sh_ann)

            path_input = '{0}input/{2}/p{1}/de_{3}_{1}.nrrd'.format(path_to_data, i, xx_name[1], xx_name[0])
            I_input = sitk.GetArrayFromImage(sitk.ReadImage(path_input))
            sh_input = I_input.shape
            if sh_ann != sh_input:
                raise Exception('size of {} != size of {}, ({} != {})'.format(path_ann, path_input, sh_ann, sh_input))

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

    print("Unique shapes ({}): {}".format(len(allShapes), allShapes))
    print("Unique YX dimensions ({}): {}".format(len(allYXDim), allYXDim))

    for yx in allYXDim:
        print(yx)
        toShow = []
        for path_input in allYXDim[yx][1]:
            I_input = sitk.GetArrayFromImage(sitk.ReadImage(path_input))
            I_input = I_input[round(I_input.shape[0] / 2)]
            I_input = I_input / np.max(I_input)

            toShow.append(I_input)

        I_out = sitk.GetImageFromArray(np.array(np.concatenate(toShow, axis=1) * 255, dtype=np.uint8))
        sitk.WriteImage(I_out, "{0}data_exploration/shape={1}.png".format(path_to_results, yx))


if __name__ == "__main__":
    main()