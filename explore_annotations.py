import SimpleITK as sitk
import numpy as np
from imshow_3D import imshow3D

kcl_nrs = [9,
29,
19,
11,
25,
18,
10,
1,
2,
4]

utah_nrs = [6,
24,
15,
16,
20,
8,
13,
27,
22,
30,
]

yale_nrs = [7,
17,
23,
26,
21,
3,
12,
14,
28,
5]

for nr in range(1, 31):
    postfix = '_b_' + str(nr)

    utah_path = 'C:/Users/cdv18/Downloads/utah_zipped/utah{}.nrrd'.format(postfix)
    kcl_path = 'C:/Users/cdv18/Downloads/kcl_zipped/kcl{}.nrrd'.format(postfix)
    yale_path = 'C:/Users/cdv18/Downloads/yale_zipped/yale{}.nrrd'.format(postfix)
    staple_path = 'C:/Users/cdv18/Documents/LAUnet/data/annotations_staple/staple{}.gipl'.format(postfix)
    mri_path = 'C:/Users/cdv18/Documents/LAUnet/data/input/post/p{}/de{}.nrrd'.format(nr, postfix)

    mri = sitk.GetArrayFromImage(sitk.ReadImage(mri_path))
    utah = sitk.GetArrayFromImage(sitk.ReadImage(utah_path))
    kcl = sitk.GetArrayFromImage(sitk.ReadImage(kcl_path))
    staple = sitk.GetArrayFromImage(sitk.ReadImage(staple_path))

    try:
        yale = sitk.GetArrayFromImage(sitk.ReadImage(yale_path))
    except RuntimeError:
        yale = np.zeros(mri.shape)

    union = ((utah + kcl + yale) > 0).astype(np.uint8)

    ann_path = 'C:/Users/cdv18/Documents/LAUnet/data/annotations/'
    out_path = '{}ann{}.nrrd'.format(ann_path, postfix)

    if nr in kcl_nrs:
        print('KCL')
    if nr in utah_nrs:
        print('UTAH')
    if nr in yale_nrs:
        print('YALE')

    # imshow3D(union)

    sitk.WriteImage(
        sitk.GetImageFromArray(
            union
        ), out_path
    )

    # imshow3D(
    #     np.concatenate(
    #         (
    #             np.concatenate(
    #                 (mri, staple * np.max(mri), kcl * np.max(mri)), axis=2
    #             ),
    #             np.concatenate(
    #                 (yale * np.max(mri), utah * np.max(mri), union * np.max(mri)), axis=2
    #             )
    #         ), axis=1
    #     )
    # )