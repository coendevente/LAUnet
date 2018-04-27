import SimpleITK as sitk
import numpy as np
from settings import *


def loadImages(pathNames):
    im_out = []
    for p in pathNames:
        im = sitk.GetArrayFromImage(sitk.ReadImage(p))
        im_out.append(im)

    return im_out


def cropImage(I, corner, dims):
    d, h, w = dims
    z, y, x = corner
    return I[z:z+d, y:y+h, x:x+w]
