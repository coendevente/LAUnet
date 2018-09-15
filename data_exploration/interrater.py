from core.settings import Settings
from core.helper_functions import Helper
import SimpleITK as sitk
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy import stats


class Test:
    def __init__(self, s, h):
        self.s = s
        self.h = h

    def calcMetrics(self, A, B):  # A is predicted, B is ground truth
        metrics = {}
        A = A.astype(bool)
        B = B.astype(bool)

        TP = np.sum(np.logical_and(A, B))
        FP = np.sum(np.logical_and(A, np.logical_not(B)))
        TN = np.sum(np.logical_and(np.logical_not(A), np.logical_not(B)))
        FN = np.sum(np.logical_and(np.logical_not(A), B))

        if 'Dice' in self.s.METRICS:
            metrics['Dice'] = 2 * TP / (2 * TP + FP + FN)
        if 'accuracy' in self.s.METRICS:
            metrics['accuracy'] = (TP + TN) / (TP + TN + FP + FN)
        if 'sensitivity' in self.s.METRICS:
            metrics['sensitivity'] = TP / (TP + FN)
        if 'specificity' in self.s.METRICS:
            metrics['specificity'] = TN / (TN + FP)
        if 'precision' in self.s.METRICS:
            metrics['precision'] = TP / (TP + FP) if TP + FP != 0 else 0
        if 'TP' in self.s.METRICS:
            metrics['TP'] = TP
        if 'FP' in self.s.METRICS:
            metrics['FP'] = FP
        if 'TN' in self.s.METRICS:
            metrics['TN'] = TN
        if 'FN' in self.s.METRICS:
            metrics['FN'] = FN
        if 'volume_diff' in self.s.METRICS:
            V_A = np.sum(A)
            V_B = np.sum(B)
            metrics['volume_diff'] = abs(V_A - V_B)

        return metrics

rater1 = []
rater2 = []

nrs = [13, 42, 25, 7, 24, 37, 40, 51, 4, 45, 48, 10, 22]

for nr in nrs:
    rater1.append(
        '/Users/coendevente/Desktop/Master/jaar_1/Stage/LAUnet/set_of_54/input/p{0}/scar_seg.nii'.format(nr)
    )
    rater2.append(
        '/Users/coendevente/Downloads/thresholded_mri done/scar_seg{}.nii'.format(nr)
    )

s = Settings()
h = Helper(s)
t = Test(s, h)

dice = []
cohen = []
for i in range(len(nrs)):
    im1 = sitk.GetArrayFromImage(sitk.ReadImage(rater1[i]))
    im2 = sitk.GetArrayFromImage(sitk.ReadImage(rater2[i]))
    metrics = t.calcMetrics(im1, im2)

    dice.append(metrics['Dice'])
    cohen.append(cohen_kappa_score(im1.flatten(), im2.flatten()))

print(dice)
print('{} \pm {}'.format(np.mean(dice), np.std(dice)))

print(cohen)
print('{} \pm {}'.format(np.mean(cohen), np.std(cohen)))

print(stats.ttest_ind(dice, dice).pvalue)