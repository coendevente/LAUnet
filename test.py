from settings import *
from helper_functions import *
import keras
from keras.models import load_model
from online_augment import OnlineAugmenter
import SimpleITK as sitk
from imshow_3D import imshow3D
import matplotlib.pyplot as plt
import tensorflow as tf
from predict import Predict


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

    # def save_metrics(self, metric_means, metric_sds, all_dice):

    def test(self):
        if self.s.CALC_PROBS:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.2)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        x_all_path, y_all_path = self.h.getImagePaths(self.s.VALTEST_SET, False)

        x_full_all = self.h.loadImages(x_all_path)
        y_full_all = self.h.loadImages(y_all_path)

        self.s.FN_CLASS_WEIGHT = 0  # Needs to be set for Keras, but is not used when testing and validating
        self.h.s = self.s
        keras.losses.custom_loss = self.h.custom_loss

        if s.USE_SE2:
            from se2unet import se2conv
            keras.layers.se2conv = se2conv

        if self.s.CALC_PROBS:
            for model_name in self.s.VALTEST_MODEL_NAMES:
                model_path = self.h.getModelPath(model_name)
                if s.USE_SE2:
                    from se2unet import se2conv
                    model = load_model(model_path, custom_objects={'se2conv': se2conv})
                else:
                    model = load_model(model_path)

                for i in range(len(x_full_all)):
                    for j in [-1] + list(range(self.s.VALTEST_AUG_NR)):
                        input = x_full_all[i]
                        anno = y_full_all[i]

                        # sitk.WriteImage(sitk.GetImageFromArray(input), 'input.nrrd')
                        # sitk.WriteImage(sitk.GetImageFromArray(anno), 'anno.nrrd')

                        if j != -1:  # No augmentation
                            input, anno = OnlineAugmenter(self.s, self.h).augment(input, anno, False, None)

                        prob = Predict(self.s, self.h).predict(input, model)

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
                    anno = sitk.ReadImage('{}anno_image_{}_{}.nii.gz'.format(predict_path, self.s.VALTEST_SET[i], j))
                    anno = sitk.GetArrayFromImage(anno)

                    prob_thresh_path = '{}prob_thresh_image_{}_{}.nii.gz'.format(predict_path, self.s.VALTEST_SET[i], j)
                    if self.s.CALC_PROB_THRESH:
                        prob = sitk.ReadImage('{}prob_image_{}_{}.nii.gz'.format(predict_path, self.s.VALTEST_SET[i],
                                                                                 j))
                        prob = sitk.GetArrayFromImage(prob)
                        # print(np.unique(prob))
                        prob_thresh = (prob > self.s.BIN_THRESH).astype(np.uint8)

                        if self.s.USE_POST_PROCESSING:
                            prob_thresh = self.h.post_process_la_seg(prob_thresh)

                        sitk.WriteImage(sitk.GetImageFromArray(prob_thresh), prob_thresh_path)
                    else:
                        prob_thresh = sitk.GetArrayFromImage(sitk.ReadImage(prob_thresh_path))

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
            print('Image nrs: {}'.format(list(self.s.VALTEST_SET)))
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
