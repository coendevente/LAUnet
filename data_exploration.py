import numpy as np
from imshow_3D import imshow3D
import SimpleITK as sitk
from itertools import chain
import matplotlib.pyplot as plt

# Get image sizes
allShapes = {}
allYXDim = {}
for xx_name in [['a', 'pre'], ['b', 'post']]:
    for i in range(1, 31):
        if i == 20 and xx_name[0] == 'a':
            continue
        # Sizes annotation and input of i == 20 are not equal
        path_ann = 'data/annotations/staple_{2}_{1}.gipl'.format(xx_name[1], i, xx_name[0], i)
        I_ann = sitk.GetArrayFromImage(sitk.ReadImage(path_ann))
        sh_ann = I_ann.shape
        print(sh_ann)

        path_input = 'data/input/{1}/p{0}/de_{2}_{0}.nrrd'.format(i, xx_name[1], xx_name[0])
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
    sitk.WriteImage(I_out, "results/data_exploration/shape={}.png".format(yx))