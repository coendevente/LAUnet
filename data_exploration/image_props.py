from core.settings import Settings
from core.helper_functions import Helper
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import copy
import SimpleITK as sitk


def mean_intensity(img, _):
    return np.mean(img)


def std_intensity(img, _):
    return np.std(img)


def min_intensity(img, _):
    return np.min(img)


def max_intensity(img, _):
    return np.max(img)


def mean_intensity_mask(img, msk):
    return np.mean(img[msk == 1])


def std_intensity_mask(img, msk):
    return np.std(img[msk == 1])


def min_intensity_mask(img, msk):
    return np.min(img[msk == 1])


def max_intensity_mask(img, msk):
    return np.max(img[msk == 1])


def img_size_x(img, _):
    return img.shape[2]


def img_size_y(img, _):
    return img.shape[1]


def img_size_z(img, _):
    return img.shape[0]


def mean_per_std_img(img, msk):
    return mean_intensity(img, msk) / std_intensity(img, msk)


def mean_per_std_mask(img, msk):
    return mean_intensity_mask(img, msk) / std_intensity_mask(img, msk)


def std_per_mean_img(img, msk):
    return std_intensity(img, msk) / mean_intensity(img, msk)


def std_per_mean_mask(img, msk):
    return std_intensity_mask(img, msk) / mean_intensity_mask(img, msk)


def dilated_mask(msk):
    return sitk.GetArrayFromImage(
        sitk.BinaryDilate(
            sitk.GetImageFromArray(msk), 10
        )
    )


def mean_dilated_mask(img, msk):
    return np.mean(img[dilated_mask(msk) == 1])


def std_dilated_mask(img, msk):
    return np.std(img[dilated_mask(msk) == 1])


if __name__ == '__main__':
    interest_nr = 22

    s = Settings()
    s.GROUND_TRUTH = 'left_atrium'
    h = Helper(s)

    explore_set = s.TESTING_SET
    print('explore_set == {}'.format(explore_set))

    img_paths, msk_paths = h.getImagePaths(explore_set, False)
    imgs = h.loadImages(img_paths)
    msks = h.loadImages(msk_paths)

    interest_idx = int(np.argwhere(np.array(explore_set) == interest_nr))

    props = {
             'Mean intensity': lambda img, msk: mean_intensity(img, msk),
             'Std intensity': lambda img, msk: std_intensity(img, msk),
             'Mean intensity mask': lambda img, msk: mean_intensity_mask(img, msk),
             'Std intensity mask': lambda img, msk: std_intensity_mask(img, msk),
             'Min intensity': lambda img, msk: min_intensity(img, msk),
             'Max intensity': lambda img, msk: max_intensity(img, msk),
             'Min intensity mask': lambda img, msk: min_intensity_mask(img, msk),
             'Max intensity mask': lambda img, msk: max_intensity_mask(img, msk),
             'Image size x': lambda img, msk: img_size_x(img, msk),
             'Image size y': lambda img, msk: img_size_y(img, msk),
             'Image size z': lambda img, msk: img_size_z(img, msk),
             # 'Mean / std image': lambda img, msk: mean_per_std_img(img, msk),
             # 'Mean / std mask': lambda img, msk: mean_per_std_mask(img, msk),
             'Std / mean image': lambda img, msk: std_per_mean_img(img, msk),
             'Std / mean mask': lambda img, msk: std_per_mean_mask(img, msk),
             'Mean dilated mask': lambda img, msk: mean_dilated_mask(img, msk),
             'Std dilated mask': lambda img, msk: std_dilated_mask(img, msk),
             }

    props_per_img = {}

    for prop_name, prop_lambda in props.items():
        print(prop_name)
        props_per_img[prop_name] = [prop_lambda(imgs[i], msks[i]) for i in range(len(imgs))]

    plt.figure()
    n = len(props_per_img.keys())
    prob_names = list(props_per_img.keys())
    grid_size = (2, 7)
    # grid_size = (1, 2)
    print(prob_names)
    for i in range(n):
        print(prob_names[i])
        plt.subplot(grid_size[0], grid_size[1], i + 1)
        other_data = copy.copy(props_per_img[prob_names[i]])
        del other_data[interest_idx]
        # print(other_data)
        other_data = np.array(other_data)
        plt.boxplot(other_data)
        median = np.median(other_data)
        upper_quartile = np.percentile(other_data, 75)
        lower_quartile = np.percentile(other_data, 25)

        iqr = upper_quartile - lower_quartile
        upper_whisker = other_data[other_data <= upper_quartile + 1.5 * iqr].max()
        lower_whisker = other_data[other_data >= lower_quartile - 1.5 * iqr].min()
        print('lower_whisker, lower_quartile, median, upper_quartile, upper_whisker = {}, {}, {}, {}, {}'.format(
            lower_whisker, lower_quartile, median, upper_quartile, upper_whisker
        ))

        print(props_per_img[prob_names[i]][interest_idx])
        # plt.scatter([1] * len(other_data), other_data, c='b', marker='x')
        plt.scatter([1], [props_per_img[prob_names[i]][interest_idx]], c='r', marker='o', alpha=.5)
        plt.title(prob_names[i])
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
    plt.savefig('../results/image_probs.png')
    # plt.show()
