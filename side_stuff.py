import numpy as np
import SimpleITK as sitk
from imshow_3D import imshow3D


def calcMetrics(A, B):  # A is predicted, B is ground truth
    metrics = {}
    A = A.astype(bool)
    B = B.astype(bool)

    TP = np.sum(np.logical_and(A, B))
    FP = np.sum(np.logical_and(A, np.logical_not(B)))
    TN = np.sum(np.logical_and(np.logical_not(A), np.logical_not(B)))
    FN = np.sum(np.logical_and(np.logical_not(A), B))

    metrics['Dice'] = 2 * TP / (2 * TP + FP + FN)

    return metrics


all_dice = []
vol_diffs = []

# for i in list(range(1, 20)) + list(range(21, 31)):
# for i in range(1):
for i in set(range(1, 31)) - set([18]):
    Ap = 'C:/Users/cdv18/Documents/LAUnet/data/annotations_improved/kcl_b_{}.nrrd'.format(i)
    Bp = 'C:/Users/cdv18/Documents/LAUnet/data/annotations/ann_b_{}.nrrd'.format(i)
    # Bp = 'C:/Users/cdv18/Downloads/kcl_zipped/kcl_b_{}.nrrd'.format(i)
    # Bp = 'C:/Users/cdv18/Downloads/utah_zipped/utah_b_{}.nrrd'.format(i)
    # Bp = 'C:/Users/cdv18/Downloads/yale_zipped/yale_b_{}.nrrd'.format(i)
    # Ap = 'C:/Users/cdv18/Downloads/yale_zipped/yale_b_21.nrrd'.format(i)
    # Bp = 'C:/Users/cdv18/Downloads/yale_zipped/yale_b_26.nrrd'.format(i)

    print('\n{}'.format(i))

    A = sitk.GetArrayFromImage(sitk.ReadImage(Ap))
    B = sitk.GetArrayFromImage(sitk.ReadImage(Bp))
    m = calcMetrics(A, B)
    all_dice.append(m['Dice'])

    print('{} {}'.format(np.sum(A), np.sum(B)))
    diff = int(np.sum(A)) - np.sum(B)
    vol_diffs.append(diff)

    print(m['Dice'])
    print(diff)

    only_A = np.logical_and(B != A, A)
    only_B = np.logical_and(B != A, B)

    # imshow3D(only_B)

    # imshow3D(np.concatenate(
    #     (A, B, only_A, only_B), axis=2
    # ))


print('\n')
print(np.mean(all_dice))
print(np.mean(vol_diffs))
print(np.sum(np.array(vol_diffs) > 0))
print(np.sum(np.array(vol_diffs) <= 0))