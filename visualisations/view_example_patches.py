import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from core.settings import Settings
from core.helper_functions import Helper

def crop(im):
    s = 20
    return im[s, 235:385, 220:370]

nr = 2
lge_path = '../data/input/post/p{0}/de_b_{0}.nrrd'.format(nr)
la_path = '../data/input/post/p{0}/la_seg_b_{0}.nrrd'.format(nr)
scar_path = '../data/annotations/kcl_b_{}.nrrd'.format(nr)

lge = sitk.GetArrayFromImage(sitk.ReadImage(lge_path))
la = sitk.GetArrayFromImage(sitk.ReadImage(la_path))
scar = sitk.GetArrayFromImage(sitk.ReadImage(scar_path))

lge_cropped = crop(lge)
la_cropped = crop(la)
scar_cropped = crop(scar)

s = Settings()
h = Helper(s)

lge_cropped, la_cropped, scar_cropped = h.normalize_multiple_list([lge_cropped, la_cropped, scar_cropped])
lge_cropped, la_cropped, scar_cropped = lge_cropped * 255, la_cropped * 255, scar_cropped * 255
lge_cropped, la_cropped, scar_cropped = lge_cropped.astype(np.uint8), \
                                        la_cropped.astype(np.uint8), \
                                        scar_cropped.astype(np.uint8)

sitk.WriteImage(sitk.GetImageFromArray(lge_cropped), 'lge.png')
sitk.WriteImage(sitk.GetImageFromArray(la_cropped), 'la.png')
sitk.WriteImage(sitk.GetImageFromArray(scar_cropped), 'scar.png')

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(lge_cropped, cmap='Greys_r')

plt.subplot(1, 3, 2)
plt.imshow(la_cropped, cmap='Greys_r')

plt.subplot(1, 3, 3)
plt.imshow(scar_cropped, cmap='Greys_r')

plt.show()
