import math
import numpy as np


class Settings:
    def __init__(self):
        # Model to train
        self.GROUND_TRUTH = 'left_atrium'  # 'left_atrium' / 'scar_fibrosis'
        self.PRE_OR_POST_NAME = 'post'  # 'post' / 'pre'
        self.PRE_OR_POST_XX = 'b'  # 'a' / 'b'
        # self.MODEL_NAME = 'la_seg_ps_480'
        # self.MODEL_NAME = 'sf_less_contrast_enh'
        # self.MODEL_NAME = 'gs_art_fraction_2/5'
        self.MODEL_NAME = 'la_seg_new_data'

        # Path to folders
        self.PATH_TO_DATA = '../data/'
        self.PATH_TO_RESULTS = '../results/'
        self.PATH_TO_MODELS = '../results/models/'
        self.PATH_TO_AUG = '../data/augmentations/'
        self.PATH_TO_ART = '../data/augmentations/artificial/'

        # Show demo images
        self.DEMO = True

        # Division of datasets
        self.ALL_NATURAL_SET = range(1, 31)
        self.TRAINING_SET = [10, 19, 30, 13, 6, 8, 17, 1, 23, 22, 4, 7, 5, 26]  # 18 left out because of orientation,
        self.VALIDATION_SET = [25, 24, 16, 2, 14, 28, 12]  # 21 left out, since duplicate of 26
        self.TESTING_SET = [20, 29, 11, 15, 27, 9, 3]

        self.YALE_NRS_POST = [7, 17, 23, 26, 21, 3, 12, 14, 28, 5, 18]

        if self.GROUND_TRUTH == 'left_atrium':
            self.TRAINING_SET = [x for x in self.TRAINING_SET if x not in self.YALE_NRS_POST]
            self.VALIDATION_SET = [x for x in self.VALIDATION_SET if x not in self.YALE_NRS_POST]
            self.TESTING_SET = [x for x in self.TESTING_SET if x not in self.YALE_NRS_POST]

            self.ALL_NATURAL_SET = list(self.ALL_NATURAL_SET) + list(range(31, 44))

            self.TRAINING_SET += [40, 39, 42, 34, 37, 32, 36]
            self.VALIDATION_SET += [31, 38, 33]
            self.TESTING_SET += [41, 43, 35]

        # Patchsize
        # PATCH_SIZE = (3, 64, 64)
        self.PATCH_SIZE = (1, 480, 480)
        # PATCH_SIZE = (1, 384, 384)
        # PATCH_SIZE = (1, 512, 512)
        # PATCH_SIZE = (3, 128, 128)
        # PATCH_SIZE = (1, 400, 400)
        self.NR_DIM = 2  # Only 2D and 3D are supported

        self.USE_PRE_PROCESSING = True

        # Training hyperparameters
        self.UNET_DEPTH = 5
        self.LEARNING_RATE = math.pow(10, -4)
        self.BATCH_SIZE = 4
        self.NR_BATCHES = 15000
        self.NR_VAL_PATCH_PER_ITER = 8
        self.POS_NEG_PATCH_PROP = .75  # with 1, all is positive, with 0 all is negative, in between values give a mix
        self.FN_CLASS_WEIGHT = 'auto'  # custom number OR 'auto'
        self.AUTO_CLASS_WEIGHT_N = 2000  # number of samples to use for the calculation of FN_CLASS_WEIGHT if it is set
        # to 'auto'
        self.EARLY_STOPPING = True
        self.PATIENTCE_ES = 2000  # Patience of early stopping
        self.DROPOUT_AT_EVERY_LEVEL = False
        self.DROPOUT = 0.3
        self.FEATURE_MAP_INC_RATE = 2.
        self.LOSS_FUNCTION = 'dice'  # 'weighted_binary_cross_entropy' OR 'dice'
        self.MAIN_OUTPUT_LOSS_WEIGHT = .8
        self.AUX_OUTPUT_LOSS_WEIGHT = .2
        self.ART_FRACTION = 0  # with 1, all is artificial, with 0 all is natural, in between values give a mix
        self.USE_ANY_SCAR_AUX = False
        self.USE_NORMALIZATION = True

        # Offline augmentation
        self.AUGMENT_ONLINE = True
        self.NR_AUG = 100

        # Testing and validation procedure
        self.SAVE_METRICS = True
        self.VALTEST_SET = self.VALIDATION_SET  # VALIDATION_SET OR TESTING_SET
        self.VALTEST_MODEL_NAMES = [self.MODEL_NAME]
        self.VALTEST_AUG_NR = 0  # Number of augmentations per image in PREDICT_SET
        # VOXEL_OVERLAP = (0, 200, 200)
        self.VOXEL_OVERLAP = (0, 32, 32)
        self.BIN_THRESH = .5  # Threshold to binarize the probability images
        self.METRICS = ['Dice', 'accuracy', 'sensitivity', 'specificity', 'precision', 'TP', 'FP', 'TN', 'FN',
                        'volume_diff']

        self.CALC_PROBS = True  # If True, the probability images will be calculated with the predict function of Keras
        # and results will be saved to the disk. If False, the probability images will be loaded from disk. An error
        # will occur if these images do not exist on the disk.

        # Data augmentation
        self.FLIP_PROB = 0

        self.ROT_MAX = 20
        self.ROT_MIN = -self.ROT_MAX

        self.ZOOM_MAX = .2
        self.ZOOM_X_MIN = 1 - self.ZOOM_MAX
        self.ZOOM_X_MAX = 1 + self.ZOOM_MAX
        self.ZOOM_Y_MIN = 1 - self.ZOOM_MAX
        self.ZOOM_Y_MAX = 1 + self.ZOOM_MAX

        self.SHEAR_MAX = .1
        self.SHEAR_X_MIN = -self.SHEAR_MAX
        self.SHEAR_X_MAX = self.SHEAR_MAX
        self.SHEAR_Y_MIN = -self.SHEAR_MAX
        self.SHEAR_Y_MAX = self.SHEAR_MAX

        self.NOISE_MEAN_MIN = 0
        self.NOISE_MEAN_MAX = 0
        self.NOISE_STD_MIN = 1
        self.NOISE_STD_MAX = 15

        self.CONTRAST_POWER_MIN = .9
        self.CONTRAST_POWER_MAX = 1.5

        # Scar applier
        self.PATH_TO_NO_SCAR_POST = '../data/input/post/'
        self.PATH_TO_NO_SCAR_PRE = '../data/input/pre/'
        self.NO_SCAR_NRS_PRE = range(1, 31)
        self.NO_SCAR_NRS_PRE = \
            [x for x in self.NO_SCAR_NRS_PRE if x not in [3, 5, 7, 9, 11, 12, 19, 20, 23, 25, 26, 28]]
        self.NO_SCAR_NRS_POST = range(1, 31)  # 18, 16, 2
        self.NO_SCAR_NRS_POST = [x for x in self.NO_SCAR_NRS_POST if x not in self.YALE_NRS_POST]

        self.NR_ART = 100
        self.PATH_TO_ARTIFICIAL_SCAR = '../data/artificial_scar/'
        self.WALL_THICKNESS_MIN_MM = 1  # mm
        self.WALL_THICKNESS_MAX_MM = 2  # mm
        self.NB_GROUPS_ODDS = [0, .2, .6, .2]
        self.ANGLE_MIN = 10
        self.ANGLE_MAX = 40
        assert abs(np.sum(self.NB_GROUPS_ODDS) - 1) < 0.00001
        self.BP_STD_FACTOR_MEAN_MIN = 2
        self.BP_STD_FACTOR_MEAN_MAX = 4
        self.BP_STD_FACTOR_STD = 1
        self.MAX_SCALE_EDGE_MM = .75  # mm
        self.SF_REMOVE_DILATION_MM = 1  # mm
        self.NOISE_RESAMPLE_FACTOR_MM = 1.3  # mm
        self.INPAINT_DISTANCE_MM = 2
        self.INPAINT_NEIGHBOURHOOD = .5
        self.BLUR_SCALE_MM = .8  # mm


if __name__ == '__main__':
    s = Settings()