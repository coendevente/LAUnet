from settings import *
from helper_functions import *
import keras
from keras.models import load_model
from augment import augment
import SimpleITK as sitk
keras.losses.custom_loss = custom_loss


def patchCornersFullImage(sh):
    step_size = np.subtract(PATCH_SIZE, VOXEL_OVERLAP)
    nr_steps = np.divide(sh, step_size)

    # Will be 1 at dimension where they are not rounded numbers, 0 otherwise
    steps_are_not_round = np.array(np.not_equal(nr_steps, np.round(nr_steps) * 1.0), dtype=np.int)
    nr_steps = (np.floor(nr_steps) - 2 - steps_are_not_round).astype(int)

    corners_dim = []
    for i in range(3):
        corners_dim[i] = np.multiply(range(nr_steps[0]), PATCH_SIZE[0])
        if steps_are_not_round[i]:
            corners_dim[i].append(sh[i] - PATCH_SIZE[i])

    patch_corners = []
    for z in corners_dim[0]:
        for y in corners_dim[1]:
            for x in corners_dim[2]:
                patch_corner = np.multiply((z, y, x), PATCH_SIZE)
                patch_corners.append(patch_corner)

    return patch_corners


def patchesFromCorners(I, patch_corners):
    patches = []
    for c in patch_corners:
        p = cropImage(I, c, PATCH_SIZE)
        patches.append(p)
    return patches


def probPatches(patches, model):
    prob_patches = []

    print(len(patches))
    cnt = 0
    for p in patches:
        print(cnt)

        p_reshaped = np.reshape(p, (1, ) + p.shape + (1, ))
        prob_p = model.predict(p_reshaped)
        prob_patches.append(prob_p)

        cnt += 1

    return prob_patches


def fullImageFromPatches(sh, prob_patches, patch_corners):
    prob_image = np.zeros(sh)
    count_image = np.zeros(sh)
    for i in range(len(patch_corners)):
        p = prob_patches[i]
        c = patch_corners[i]

        prob_image[c[0]:c[0]+PATCH_SIZE[0], c[1]:c[1]+PATCH_SIZE[1], c[2]:c[2]+PATCH_SIZE[2]] += p
        count_image[c[0]:c[0]+PATCH_SIZE[0], c[1]:c[1]+PATCH_SIZE[1], c[2]:c[2]+PATCH_SIZE[2]] += 1
    prob_image /= count_image
    return prob_image


def probImage(I, model):
    patch_corners = patchCornersFullImage(I.shape)
    patches = patchesFromCorners(I, patch_corners)
    prob_patches = probPatches(patches, model)
    prob_image = fullImageFromPatches(I.shape, prob_patches, patch_corners)
    return prob_image


def calcMetrics(A, B):  # A is predicted, B is ground truth
    metrics = {}
    A = A.astype(bool)
    B = B.astype(bool)

    TP = np.sum(np.logical_and(A, B))
    FP = np.sum(np.logical_and(A, not B))
    TN = np.sum(np.logical_and(not A, not B))
    FN = np.sum(np.logical_and(A, not B))

    if 'Dice' in METRICS:
        metrics['Dice'] = 2 * TP / (2 * TP + FP + FN)
    if 'accuracy' in METRICS:
        metrics['accuracy'] = (TP + TN) / (TP + TN + FP + FN)
    if 'sensitivity' in METRICS:
        metrics['sensitivity'] = TP / (TP + TN)
    if 'specificity' in METRICS:
        metrics['specificity'] = TN / (TN + FP)
    if 'precision' in METRICS:
        metrics['precision'] = TN / (TP + FP)

    return metrics


def main():
    x_all_path, y_all_path = getImagePaths(VALTEST_SET)

    x_full_all = loadImages(x_all_path)
    y_full_all = loadImages(y_all_path)

    if CALC_PROBS:
        for model_name in VALTEST_MODEL_NAMES:
            model_path = getModelPath(model_name)
            model = load_model(model_path)

            for i in range(len(x_full_all)):
                for j in [-1] + list(range(VALTEST_AUG_NR)):
                    input = x_full_all[i]
                    anno = y_full_all[i]

                    if j != -1:  # No augmentation
                        input, anno = augment(input, anno)

                    prob = probImage(input, model)

                    predict_path = getModelPredictPath(model_name)
                    sitk.WriteImage(input, '{}input_image_{}_{}.nrrd'.format(predict_path, i, j))
                    sitk.WriteImage(prob, '{}prob_image_{}_{}.nrrd'.format(predict_path, i, j))
                    sitk.WriteImage(anno, '{}anno_image_{}_{}.nrrd'.format(predict_path, i, j))

    # Calculate the metrics
    all_metrics = {}
    metric_means = {}
    metric_sds = {}

    for model_name in VALTEST_MODEL_NAMES:
        all_metrics[model_name] = []

        for i in range(len(x_full_all)):
            for j in [-1] + list(range(VALTEST_AUG_NR)):
                predict_path = getModelPredictPath(model_name)
                prob = sitk.ReadImage('{}prob_image_{}_{}.nrrd'.format(predict_path, i, j))
                anno = sitk.ReadImage('{}anno_image_{}_{}.nrrd'.format(predict_path, i, j))

                prob_thresh = prob > BIN_THRESH
                metrics = calcMetrics(prob_thresh, anno)
                all_metrics[model_name].append(metrics)

        # Calculate averages of metrics
        metric_means[model_name] = {}
        metric_sds[model_name] = {}
        for metric in all_metrics[model_name].items():
            all = [all_metrics[model_name][i][metric] for i in len(all_metrics[-1])]
            metric_means[model_name][metric] = np.mean(all)
            metric_sds[model_name][metric] = np.std(all)

    print(metric_means)
    print(metric_sds)


if __name__ == "__main__":
    main()