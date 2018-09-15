import sys
sys.path.append("./")

from core.helper_functions import *
import keras
from keras.models import load_model
from core.augmentations.online_augment import OnlineAugmenter
import SimpleITK as sitk
from core.predict import Predict


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
        x_all_path, y_all_path = self.h.getImagePaths(self.s.VALTEST_SET, False)

        x_full_all = self.h.loadImages(x_all_path)
        if self.s.USE_ANNO_FOR_TEST:
            y_full_all = self.h.loadImages(y_all_path)

        self.s.FN_CLASS_WEIGHT = 0  # Needs to be set for Keras, but is not used when testing and validating
        self.h.s = self.s
        keras.losses.custom_loss = self.h.custom_loss

        ext = 'nii.gz'

        if self.s.USE_SE2:
            from core.architectures.se2unet import se2conv
            keras.layers.se2conv = se2conv

        if self.s.CALC_PROBS:
            for model_name in self.s.VALTEST_MODEL_NAMES:
                model_path = self.h.getModelPath(model_name)
                if self.s.USE_SE2:
                    from core.architectures.se2unet import se2conv
                    model = load_model(model_path, custom_objects={'se2conv': se2conv})
                else:
                    model = load_model(model_path)

                for i in range(len(x_full_all)):
                    for j in [-1] + list(range(self.s.VALTEST_AUG_NR)):
                        input = x_full_all[i]
                        if self.s.USE_ANNO_FOR_TEST:
                            anno = y_full_all[i]

                        # sitk.WriteImage(sitk.GetImageFromArray(input), 'input.nrrd')
                        # sitk.WriteImage(sitk.GetImageFromArray(anno), 'anno.nrrd')

                        if j != -1:  # No augmentation
                            input, anno = OnlineAugmenter(self.s, self.h).augment(input, anno, False, None)

                        predict = Predict(self.s, self.h)
                        prob = predict.predict(input, model)
                        del predict

                        predict_path = self.h.getModelPredictPath(model_name, self.s.PREDICT_AUX_OUTPUT)

                        sitk.WriteImage(sitk.GetImageFromArray(input),
                                        '{}input_image_{}_{}.{}'.format(predict_path, self.s.VALTEST_SET[i], j, ext))

                        sitk.WriteImage(sitk.GetImageFromArray(prob),
                                        '{}prob_image_{}_{}.{}'.format(predict_path, self.s.VALTEST_SET[i], j, ext))

                        if self.s.USE_ANNO_FOR_TEST:
                            sitk.WriteImage(sitk.GetImageFromArray(anno),
                                            '{}anno_image_{}_{}.{}'.format(predict_path, self.s.VALTEST_SET[i], j, ext))
                        print("Saved in {}".format(predict_path))

        # Calculate the metrics
        all_metrics = {}
        metric_means = {}
        metric_sds = {}

        for model_name in self.s.VALTEST_MODEL_NAMES:
            all_metrics[model_name] = []

            for i in range(len(x_full_all)):
                for j in [-1] + list(range(self.s.VALTEST_AUG_NR)):
                    predict_path = self.h.getModelPredictPath(model_name, self.s.PREDICT_AUX_OUTPUT)

                    if self.s.USE_ANNO_FOR_TEST:
                        anno = sitk.ReadImage('{}anno_image_{}_{}.{}'.format(predict_path, self.s.VALTEST_SET[i], j,
                                                                             ext))
                        anno = sitk.GetArrayFromImage(anno)

                    prob_thresh_path = '{}prob_thresh_image_{}_{}.{}'.format(predict_path, self.s.VALTEST_SET[i], j,
                                                                             ext)
                    if self.s.CALC_PROB_THRESH:
                        prob = sitk.ReadImage('{}prob_image_{}_{}.{}'.format(predict_path, self.s.VALTEST_SET[i], j,
                                                                             ext))
                        prob = sitk.GetArrayFromImage(prob)
                        # print(np.unique(prob))
                        prob_thresh = (prob > self.s.BIN_THRESH).astype(np.uint8)

                        if self.s.USE_POST_PROCESSING:
                            prob_thresh = self.h.post_process_la_seg(prob_thresh)

                        sitk.WriteImage(sitk.GetImageFromArray(prob_thresh), prob_thresh_path)
                    else:
                        prob_thresh = sitk.GetArrayFromImage(sitk.ReadImage(prob_thresh_path))

                    if self.s.DISCARD_LAST_SLICE:
                        prob_thresh[-1, :, :] = 0
                        sitk.WriteImage(sitk.GetImageFromArray(prob_thresh), prob_thresh_path)

                    if self.s.USE_ANNO_FOR_TEST:
                        metrics = self.calcMetrics(prob_thresh, anno)
                        all_metrics[model_name].append(metrics)

            if self.s.USE_ANNO_FOR_TEST:
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
