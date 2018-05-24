from settings import *
from helper_functions import *
import keras
from keras.models import load_model
from online_augment import OnlineAugmenter
import SimpleITK as sitk
from imshow_3D import imshow3D
import matplotlib.pyplot as plt
import tensorflow as tf


class Test:
    def __init__(self, s, h):
        self.s = s
        self.h = h

    def patchCornersFullImage(self, sh):
        step_size = np.subtract(self.s.PATCH_SIZE, self.s.VOXEL_OVERLAP)
        nr_steps = np.divide(sh, step_size)

        # Will be 1 at dimension where they are not rounded numbers, 0 otherwise
        steps_are_not_round = np.array(np.not_equal(nr_steps, np.round(nr_steps) * 1.0), dtype=np.int)
        nr_steps = (np.floor(nr_steps) - steps_are_not_round).astype(int)

        corners_dim = []
        for i in range(3):
            corners_dim.append(np.array(range(nr_steps[i] + 1)) * step_size[i])
            if steps_are_not_round[i]:
                corners_dim[i] = np.append(corners_dim[i], sh[i] - self.s.PATCH_SIZE[i])

            for j in reversed(range(corners_dim[i].shape[0])):
                if corners_dim[i][j] + self.s.PATCH_SIZE[i] > sh[i]:
                    print("i, j == {}, {}".format(i, j))
                    corners_dim[i] = np.delete(corners_dim[i], j)

        patch_corners = []
        for z in corners_dim[0]:
            for y in corners_dim[1]:
                for x in corners_dim[2]:
                    patch_corner = (z, y, x)  # np.multiply((z, y, x), PATCH_SIZE)
                    patch_corners.append(patch_corner)

        return patch_corners

    def patchesFromCorners(self, I, patch_corners):
        patches = []
        for c in patch_corners:
            if self.s.PATCH_SIZE[1] <= I.shape[1]:
                p = self.h.cropImage(I, c, self.s.PATCH_SIZE)
            elif self.s.PATCH_SIZE[1] > I.shape[1]:
                p = self.h.rescaleImage(I[c[0]:c[0] + self.s.PATCH_SIZE[0]], self.s.PATCH_SIZE[1:])
            patches.append(p)
        return patches

    def probPatches(self, patches, model):
        prob_patches = []

        print(len(patches))
        cnt = 0
        for p in patches:
            if cnt % 1 == 0:
                print(cnt)

            ps = p.shape
            if self.s.NR_DIM == 2:
                ps = ps[1:]

            p_reshaped = np.reshape(p, (1, ) + ps + (1, ))

            if self.s.USE_NORMALIZATION:
                p_reshaped = self.h.normalize(p_reshaped)

            prob_p = model.predict(p_reshaped)

            if self.s.USE_ANY_SCAR_AUX:
                prob_p = prob_p[0]

            prop_p_s = prob_p.shape[1:4]
            if self.s.NR_DIM == 2:
                prop_p_s = (1, ) + prob_p.shape[1:3]
            prob_p_reshaped = np.reshape(prob_p, prop_p_s)
            prob_patches.append(prob_p_reshaped)

            cnt += 1

        return prob_patches

    def fullImageFromPatches(self, sh, prob_patches, patch_corners):
        prob_image = np.zeros(sh)
        count_image = np.zeros(sh)

        for i in range(len(patch_corners)):
            p = prob_patches[i]
            c = list(patch_corners[i])

            for j in range(len(c)):
                if c[j] < 0:
                    c[j] = 0

            # print('c == {}'.format(c))

            if prob_image.shape[1] < p.shape[1]:
                p = self.h.rescaleImage(p, prob_image.shape[1:])

            # print(prob_image.shape)
            # print(p.shape)

            prob_image[c[0]:c[0] + self.s.PATCH_SIZE[0],
                       c[1]:c[1] + self.s.PATCH_SIZE[1],
                       c[2]:c[2] + self.s.PATCH_SIZE[2]] += p

            count_image[c[0]:c[0] + self.s.PATCH_SIZE[0],
                        c[1]:c[1] + self.s.PATCH_SIZE[1],
                        c[2]:c[2] + self.s.PATCH_SIZE[2]] += 1

        # imshow3D(count_image)
        prob_image /= count_image
        return prob_image

    def probImage(self, I, model):
        patch_corners = self.patchCornersFullImage(I.shape)
        patches = self.patchesFromCorners(I, patch_corners)
        prob_patches = self.probPatches(patches, model)
        prob_image = self.fullImageFromPatches(I.shape, prob_patches, patch_corners)
        return prob_image

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

    # def save_metrics(self, metric_means, metric_sds, all_dice):

    def test(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        x_all_path, y_all_path = self.h.getImagePaths(self.s.VALTEST_SET, False)

        x_full_all = self.h.loadImages(x_all_path)
        y_full_all = self.h.loadImages(y_all_path)

        self.s.FN_CLASS_WEIGHT = 0  # Needs to be set for Keras, but is not used when testing and validating
        self.h.s = self.s
        keras.losses.custom_loss = self.h.custom_loss

        if self.s.CALC_PROBS:
            for model_name in self.s.VALTEST_MODEL_NAMES:
                model_path = self.h.getModelPath(model_name)
                model = load_model(model_path)

                for i in range(len(x_full_all)):
                    for j in [-1] + list(range(self.s.VALTEST_AUG_NR)):
                        input = x_full_all[i]
                        anno = y_full_all[i]

                        if input.shape[1] < self.s.PATCH_SIZE[1]:
                            input = self.h.rescaleImage(input, self.s.PATCH_SIZE[1:])
                            anno = (self.h.rescaleImage(anno, self.s.PATCH_SIZE[1:]) > 0).astype(np.uint8)

                        # sitk.WriteImage(sitk.GetImageFromArray(input), 'input.nrrd')
                        # sitk.WriteImage(sitk.GetImageFromArray(anno), 'anno.nrrd')

                        if j != -1:  # No augmentation
                            input, anno = OnlineAugmenter(self.s, self.h).augment(input, anno, False)

                        prob = self.probImage(input, model)

                        predict_path = self.h.getModelPredictPath(model_name)
                        sitk.WriteImage(sitk.GetImageFromArray(input),
                                        '{}input_image_{}_{}.nii.gz'.format(predict_path, self.s.VALTEST_SET[i], j))

                        sitk.WriteImage(sitk.GetImageFromArray(prob),
                                        '{}prob_image_{}_{}.nii.gz'.format(predict_path, self.s.VALTEST_SET[i], j))

                        sitk.WriteImage(sitk.GetImageFromArray(anno),
                                        '{}anno_image_{}_{}.nii.gz'.format(predict_path, self.s.VALTEST_SET[i], j))
                        print("Saved in {}".format(predict_path))

        # Calculate the metrics
        all_metrics = {}
        metric_means = {}
        metric_sds = {}

        for model_name in self.s.VALTEST_MODEL_NAMES:
            all_metrics[model_name] = []

            for i in range(len(x_full_all)):
                for j in [-1] + list(range(self.s.VALTEST_AUG_NR)):
                    predict_path = self.h.getModelPredictPath(model_name)
                    prob = sitk.ReadImage('{}prob_image_{}_{}.nii.gz'.format(predict_path, self.s.VALTEST_SET[i], j))
                    anno = sitk.ReadImage('{}anno_image_{}_{}.nii.gz'.format(predict_path, self.s.VALTEST_SET[i], j))

                    prob = sitk.GetArrayFromImage(prob)
                    anno = sitk.GetArrayFromImage(anno)

                    # print(np.unique(prob))
                    prob_thresh = prob > self.s.BIN_THRESH

                    predict_path = self.h.getModelPredictPath(model_name)
                    sitk.WriteImage(sitk.GetImageFromArray(prob_thresh.astype(np.uint16)),
                                    '{}prob_thresh_image_{}_{}.nii.gz'.format(predict_path, self.s.VALTEST_SET[i], j))

                    metrics = self.calcMetrics(prob_thresh, anno)
                    all_metrics[model_name].append(metrics)

            # Calculate averages of metrics
            metric_means[model_name] = {}
            metric_sds[model_name] = {}
            for metric in all_metrics[model_name][0].items():
                all = [all_metrics[model_name][i][metric[0]] for i in range(len(all_metrics[model_name]))]
                metric_means[model_name][metric[0]] = np.mean(all)
                metric_sds[model_name][metric[0]] = np.std(all)

            print('=========== Results of {} ==========='.format(model_name))
            print('Means of metrics: {}'.format(metric_means[model_name]))
            print('Standard deviations of metrics: {}'.format(metric_sds[model_name]))

        all_dice = [all_metrics[model_name][i]['Dice'] for i in range(len(all_metrics[model_name]))]
        print('All Dice values: {}'.format(all_dice))

        # plt.figure()
        # plt.hist(all_dice)
        # plt.show()

        return metric_means, metric_sds


if __name__ == "__main__":
    s = Settings()
    h = Helper(s)
    t = Test(s, h)
    _, _, = t.test()
