import glob
import SimpleITK as sitk
import numpy as np
import os
from imshow_3D import imshow3D
from skimage.transform import rescale, resize, downscale_local_mean


def rescale_image(imageA, imageC):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(imageC)
    resample.SetOutputDirection(imageA.GetDirection())
    resample.SetOutputOrigin(imageA.GetOrigin())
    imageCA = resample.Execute(imageC)

    return imageCA


good_nrs = [2, 4, 5, 6, 8, 10, 14, 18, 20, 21, 22, 24, 26]
# good_nrs = range(1, 21)
out_nrs = range(31, 31 + len(good_nrs))

path_from = '../data/extra/'

for i in range(len(good_nrs)):
    folder = 'C:/Users/cdv18/Desktop/temp/temp/case{}/'.format(good_nrs[i])
    lge_path = '{}lge.gipl'.format(folder)
    la_path = '{}LA-reg_in_lge.gipl'.format(folder)

    lge = sitk.ReadImage(lge_path)
    la = sitk.ReadImage(la_path)

    input_folder_out = '../data/input/post/p{}/'.format(out_nrs[i])
    sf_ann_folder_out = '../data/annotations/'

    if lge.GetSize() != la.GetSize():
        print('For {}: {} != {}'.format(good_nrs[i], lge.GetSize(), la.GetSize()))
        # la_new = sitk.GetArrayFromImage(la)

        # x, y, z = lge.GetSize()
        # la_new = rescale(la_new, )
        # la_new = sitk.GetImageFromArray(la_new)
        # la_new.CopyInformation(lge)
        # la = la_new
        # la = rescale_image(lge, la)
        #
        # imshow3D(sitk.GetArrayFromImage(la))
        #
        # print('For {}: {} == {}'.format(good_nrs[i], lge.GetSize(), la.GetSize()))

        assert lge.GetSize() == la.GetSize()

    if not os.path.exists(input_folder_out):
        os.makedirs(input_folder_out)

    lge_path_out = '{}de_b_{}.nrrd'.format(input_folder_out, out_nrs[i])
    la_path_out = '{}la_seg_b_{}.nrrd'.format(input_folder_out, out_nrs[i])
    sf_ann_path_out = '{}kcl_b_{}.nrrd'.format(sf_ann_folder_out, out_nrs[i])

    print(reversed(lge.GetSize()))
    sf_ann = sitk.GetImageFromArray(
        np.zeros(
            list(
                reversed(
                    lge.GetSize()
                )
            )
        )
    )
    sf_ann.CopyInformation(lge)

    sitk.WriteImage(lge, lge_path_out)
    sitk.WriteImage(la, la_path_out)
    sitk.WriteImage(sf_ann, sf_ann_path_out)
