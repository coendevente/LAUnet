import numpy as np
import SimpleITK as sitk


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

Ap = '/Users/coendevente/Desktop/Master/jaar_1/Stage/LAUnet/data/annotations_improved/kcl_b_1.nrrd'

all_dice = []

# for i in list(range(1, 20)) + list(range(21, 31)):
for i in range(1, 31):
    Ap = '/Users/coendevente/Downloads/utah_zipped/utah_b_{}.nrrd'.format(i)
    Bp = '/Users/coendevente/Downloads/kcl_zipped 2/kcl_b_{}.nrrd'.format(i)
    Cp = '/Users/coendevente/Downloads/yale_zipped/yale_b_{}.nrrd'.format(i)

    A = sitk.GetArrayFromImage(sitk.ReadImage(Ap))
    B = sitk.GetArrayFromImage(sitk.ReadImage(Bp))
    try:
        C = sitk.GetArrayFromImage(sitk.ReadImage(Cp))
    except Exception:
        a = 0

    m = calcMetrics(A, B)
    all_dice.append(m['Dice'])

print(all_dice)
print(np.mean(all_dice))