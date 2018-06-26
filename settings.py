import math
import numpy as np
import matplotlib.pyplot as plt
import platform


class Settings:
    def __init__(self):
        # Model to train
        self.GROUND_TRUTH = 'scar_fibrosis'  # 'left_atrium' / 'scar_fibrosis'
        self.PRE_OR_POST_NAME = 'post'  # 'post' / 'pre'
        self.PRE_OR_POST_XX = 'b'  # 'a' / 'b'
        self.MODEL_NAME = 'sf_la_input'

        self.DATA_SET = 'original'  # 'original' OR 'challenge_2018'

        # Path to folders
        self.DATA_PRE = '/data/cwdevente/LAUnet/' if platform.system() == 'Linux' else '../'
        self.PATH_TO_DATA = '{}challenge_2018_data/'.format(self.DATA_PRE) \
            if self.DATA_SET == 'challenge_2018' else '{}data/'.format(self.DATA_PRE)
        self.PATH_TO_RESULTS = '../results/'
        self.PATH_TO_MODELS = '../results/models/'
        self.PATH_TO_AUG = self.PATH_TO_DATA + 'augmentations/'
        self.PATH_TO_ART = self.PATH_TO_DATA + 'augmentations/artificial/'
        # print(self.DATA_PRE)

        # Show demo images
        self.DEMO = True

        self.YALE_NRS_POST = [7, 17, 23, 26, 21, 3, 12, 14, 28, 5, 18]

        if self.DATA_SET == 'original':
            # Division of datasets
            self.ALL_NATURAL_SET = range(1, 31)
            self.TRAINING_SET = [10, 19, 30, 13, 6, 8, 17, 1, 23, 22, 4, 7, 5, 26]  # 18 left out because of orientation
            self.VALIDATION_SET = [25, 24, 16, 2, 14, 28, 12]  # 21 left out, since duplicate of 26
            self.TESTING_SET = [20, 29, 11, 15, 27, 9, 3]

            if self.GROUND_TRUTH == 'left_atrium':
                self.ALL_NATURAL_SET = [x for x in self.ALL_NATURAL_SET if x not in self.YALE_NRS_POST]
                self.TRAINING_SET = [x for x in self.TRAINING_SET if x not in self.YALE_NRS_POST]
                self.VALIDATION_SET = [x for x in self.VALIDATION_SET if x not in self.YALE_NRS_POST]
                self.TESTING_SET = [x for x in self.TESTING_SET if x not in self.YALE_NRS_POST]

                # self.ALL_NATURAL_SET = list(self.ALL_NATURAL_SET) + list(range(31, 44))
                # self.ALL_NATURAL_SET = list(range(31, 44))
                #
                # self.TRAINING_SET += [40, 39, 42, 34, 37, 32, 36]
                # self.VALIDATION_SET += [31, 38, 33]
                # self.TESTING_SET += [41, 43, 35]

                # print(self.VALIDATION_SET)
        elif self.DATA_SET == 'challenge_2018':
            self.ALL_NATURAL_SET = range(1, 101)
            # self.TRAINING_SET = range(1, 51)
            # self.VALIDATION_SET = range(51, 76)
            # self.TESTING_SET = range(76, 101)

            np.random.seed(0)
            r_p = np.random.permutation(self.ALL_NATURAL_SET)
            self.TRAINING_SET = r_p[range(0, 50)]
            self.VALIDATION_SET = r_p[range(50, 75)]
            self.TESTING_SET = r_p[range(75, 100)]

        # Patchsize
        self.VARIABLE_PATCH_SIZE = False

        # self.PATCH_SIZE = (3, 64, 64)
        # self.PATCH_SIZE = (1, 480, 480)
        self.PATCH_SIZE = (1, 448, 448)
        self.NR_DIM = 2  # Only 2D and 3D are supported

        self.USE_PRE_PROCESSING = False

        # Training hyperparameters
        self.USE_SE2 = True
        self.UNET_DEPTH = 4
        self.LEARNING_RATE = math.pow(10, -3)
        self.NR_CONV_PER_CONV_BLOCK = 2
        self.BATCH_SIZE = 4
        self.NR_BATCHES = 30000
        self.NR_VAL_PATCH_PER_ITER = 4
        self.POS_NEG_PATCH_PROP = .5  # with 1, all is positive, with 0 all is negative, in between values give a mix
        self.FN_CLASS_WEIGHT = 'auto'  # custom number OR 'auto'
        self.AUTO_CLASS_WEIGHT_N = 4000  # number of samples to use for the calculation of FN_CLASS_WEIGHT if it is set
        # to 'auto'
        self.EARLY_STOPPING = True
        self.PATIENTCE_ES = 4000  # Patience of early stopping
        self.DROPOUT_AT_EVERY_LEVEL = False
        self.DROPOUT = 0
        self.FEATURE_MAP_INC_RATE = 2.
        self.LOSS_FUNCTION = 'dice'  # 'weighted_binary_cross_entropy' OR 'dice'
        self.MAIN_OUTPUT_LOSS_WEIGHT = .8
        self.AUX_OUTPUT_LOSS_WEIGHT = .2
        self.ART_FRACTION = 0  # with 1, all is artificial, with 0 all is natural, in between values give a mix
        self.USE_ANY_SCAR_AUX = False
        self.USE_NORMALIZATION = True
        self.USE_LA_INPUT = True
        self.VAL_LOSS_SMOOTH_WINDOW_MODEL_SELECTION = 50
        self.START_CH = 16
        self.SE2_N_THETA = 8
        self.LOAD_MODEL = False  # Continue training with model file that already exists for model name

        # Offline augmentation
        self.AUGMENT_ONLINE = False
        self.NR_AUG = 100

        if self.USE_LA_INPUT and self.AUGMENT_ONLINE:
            raise Exception('USE_LA_INPUT with AUGMENT_ONLINE is not yet implemented')

        if self.USE_LA_INPUT and self.ART_FRACTION > 0:
            raise Exception('USE_LA_INPUT with artificial data is not yet implemented')

        if self.USE_LA_INPUT and self.GROUND_TRUTH == 'left_atrium':
            raise Exception('Should not be using USE_LA_INPUT with GROUND_TRUTH == \'left_atrium\'')

        # Testing and validation procedure
        self.USE_POST_PROCESSING = True
        self.SAVE_METRICS = True
        self.VALTEST_SET = self.VALIDATION_SET  #  self.TESTING_SET  # VALIDATION_SET OR TESTING_SET
        self.VALTEST_MODEL_NAMES = [self.MODEL_NAME]
        # self.VALTEST_MODEL_NAMES = ['la_2018_challenge_convpl_depth_2/{}'.format(i) for i in range(1, 7)]
        self.VALTEST_AUG_NR = 0  # Number of augmentations per image in PREDICT_SET
        # VOXEL_OVERLAP = (0, 200, 200)
        self.VOXEL_OVERLAP = (0, 32, 32)
        self.BIN_THRESH = .5  # Threshold to binarize the probability images
        self.METRICS = ['Dice', 'accuracy', 'sensitivity', 'specificity', 'precision', 'TP', 'FP', 'TN', 'FN',
                        'volume_diff']

        self.CALC_PROBS = False  # If True, the probability images will be calculated with the predict function of Keras
        # and results will be saved to the disk. If False, the probability images will be loaded from disk. An error
        # will occur if these images do not exist on the disk.
        self.CALC_PROB_THRESH = True
        self.RESIZE_BEFORE_PREDICTION = (576, 576)  # EITHER False or nD tuple where n == self.NR_DIM

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
        self.MODEL_NAME_FOR_LA_SEG = 'la_2018_challenge_convpl_depth_2/2'
        self.MODEL_PS_FOR_LA_SEG = (1, 448, 448)

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