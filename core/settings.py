import math
import numpy as np
import matplotlib.pyplot as plt
import platform


class Settings:
    def __init__(self):
        # Model to train
        self.GROUND_TRUTH = 'left_atrium'  # 'left_atrium' / 'scar_fibrosis'
        self.PRE_OR_POST_NAME = 'post'  # 'post' / 'pre'
        self.PRE_OR_POST_XX = 'b'  # 'a' / 'b'
        self.MODEL_NAME = 'la_all_100_data_do25'

        self.DATA_SET = 'atriaseg_2018_testing'  # 'new_lge_data_2' OR 'data_july_2018' OR 'original'
        # OR 'set_of_54' OR 'atriaseg_2018_testing' OR 'challenge_2018'

        # Path to folders
        self.DATA_PRE = '/data/cwdevente/LAUnet/' if platform.system() == 'Linux' else '../'

        if self.DATA_SET == 'challenge_2018':
            self.PATH_TO_DATA = '{}challenge_2018_data/'.format(self.DATA_PRE)
        elif self.DATA_SET == 'data_july_2018':
            self.PATH_TO_DATA = '{}data_july_2018/'.format(self.DATA_PRE)
        elif self.DATA_SET == 'new_lge_data_2':
            self.PATH_TO_DATA = '{}new_lge_data_2/'.format(self.DATA_PRE)
        elif self.DATA_SET == 'set_of_54':
            self.PATH_TO_DATA = '{}set_of_54/'.format(self.DATA_PRE)
        elif self.DATA_SET == 'atriaseg_2018_testing':
            self.PATH_TO_DATA = '{}atriaseg_2018_testing/'.format(self.DATA_PRE)
        else:
            self.PATH_TO_DATA = '{}data/'.format(self.DATA_PRE)

        self.PATH_TO_RESULTS = '../results/'
        self.PATH_TO_MODELS = '../results/models/'
        # self.PATH_TO_MODELS = '/data/cwdevente/old_models/'
        self.PATH_TO_AUG = self.PATH_TO_DATA + 'augmentations/'
        self.PATH_TO_ART = self.PATH_TO_DATA + 'augmentations/artificial/'
        # print(self.DATA_PRE)

        # Show demo images
        self.DEMO = False

        self.YALE_NRS_POST = [7, 17, 23, 26, 21, 3, 12, 14, 28, 5, 18]

        if self.DATA_SET == 'original':
            # Division of datasets
            # self.ALL_NATURAL_SET = list(range(1, 31))
            # self.TRAINING_SET = [10, 19, 30, 13, 6, 8, 17, 1, 23, 22, 4, 7, 5, 26]  # 18 left out because of orientation
            # self.VALIDATION_SET = [25, 24, 16, 2, 14, 28, 12]  # 21 left out, since duplicate of 26
            # self.TESTING_SET = [20, 29, 11, 15, 27, 9, 3]
            #
            # if False:
            #     a = list(range(31, 55))
            #     self.ALL_NATURAL_SET += a
            #
            #     np.random.seed(0)
            #     r_p = np.random.permutation(a)
            #     self.TRAINING_SET += list(r_p[range(0, 12)])
            #     self.VALIDATION_SET += list(r_p[range(12, 18)])
            #     self.TESTING_SET += list(r_p[range(18, 24)])
            #
            # if self.GROUND_TRUTH == 'left_atrium':
            #     self.ALL_NATURAL_SET = [x for x in self.ALL_NATURAL_SET if x not in self.YALE_NRS_POST]
            #     self.TRAINING_SET = [x for x in self.TRAINING_SET if x not in self.YALE_NRS_POST]
            #     self.VALIDATION_SET = [x for x in self.VALIDATION_SET if x not in self.YALE_NRS_POST]
            #     self.TESTING_SET = [x for x in self.TESTING_SET if x not in self.YALE_NRS_POST]

            self.ALL_NATURAL_SET = list(range(1, 31))
            self.TRAINING_SET = [1, 2, 4, 6, 8, 9, 10, 13]
            self.VALIDATION_SET = [15, 16, 19, 20, 11]
            self.TESTING_SET = [24, 25, 27, 29, 30]
        elif self.DATA_SET == 'challenge_2018':
            self.ALL_NATURAL_SET = range(1, 101)
            # self.TRAINING_SET = range(1, 51)
            # self.VALIDATION_SET = range(51, 76)
            # self.TESTING_SET = range(76, 101)

            np.random.seed(0)
            r_p = np.random.permutation(self.ALL_NATURAL_SET)
            self.TRAINING_SET = r_p[range(0, 75)]
            self.VALIDATION_SET = r_p[range(75, 100)]
            self.TESTING_SET = np.array([])
            # self.TESTING_SET = r_p[range(75, 100)]
        elif self.DATA_SET == 'atriaseg_2018_testing':
            self.ALL_NATURAL_SET = range(1, 55)
            self.TRAINING_SET = []
            self.VALIDATION_SET = []
            self.TESTING_SET = self.ALL_NATURAL_SET
        elif self.DATA_SET == 'data_july_2018':
            self.ALL_NATURAL_SET = range(1, 25)

            np.random.seed(0)
            r_p = np.random.permutation(self.ALL_NATURAL_SET)
            self.TRAINING_SET = r_p[range(0, 12)]
            self.VALIDATION_SET = r_p[range(12, 18)]
            self.TESTING_SET = r_p[range(18, 24)]
        elif self.DATA_SET == 'set_of_54':
            self.ALL_NATURAL_SET = range(1, 55)

            np.random.seed(0)

            self.TRAINING_SET = [54, 34, 49, 27, 12, 3, 33, 43, 46, 31, 5, 11, 29, 23, 32, 39, 38, 8, 15, 28, 36, 50,
                                 19, 52, 35, 16]
            self.VALIDATION_SET = [6, 17, 47, 21, 53, 9, 14, 26, 41, 44, 2, 1]  # 18, 30
            self.TESTING_SET = [13, 42, 25, 7, 24, 37, 40, 51, 4, 45, 48, 10, 22]  # 20

        # Patch size
        self.VARIABLE_PATCH_SIZE = False

        # self.PATCH_SIZE = (3, 64, 64)
        # self.PATCH_SIZE = (1, 480, 480)
        self.PATCH_SIZE = (1, 448, 448)
        self.NR_DIM = 2  # Only 2D and 3D are supported

        self.USE_PRE_PROCESSING = False

        # Training hyperparameters
        self.USE_SE2 = False
        self.UNET_DEPTH = 5
        self.LEARNING_RATE = math.pow(10, -4)
        self.NR_CONV_PER_CONV_BLOCK = 1
        self.BATCH_SIZE = 4
        self.NR_BATCHES = 60000
        self.VALIDATE_EVERY_ITER = 25
        self.NR_VAL_PATCH_PER_ITER = 100
        self.BATCH_SIZE_VAL = 4
        self.POS_NEG_PATCH_PROP = .5  # with 1, all is positive, with 0 all is negative, in between values give a mix
        self.FN_CLASS_WEIGHT = 'auto'  # custom number OR 'auto'
        self.AUTO_CLASS_WEIGHT_N = 4000  # number of samples to use for the calculation of FN_CLASS_WEIGHT if it is set
        # to 'auto'
        self.EARLY_STOPPING = True
        self.PATIENCE_ES = 4000  # Patience of early stopping
        self.RESET_PATIENCE_ES = True
        self.RESET_VAL_LOSS = True
        self.DROPOUT_AT_EVERY_LEVEL = False
        self.DROPOUT = 0.5
        self.FEATURE_MAP_INC_RATE = 2.
        self.LOSS_FUNCTION = 'dice'  # 'weighted_binary_cross_entropy' OR 'dice'
        self.MAIN_OUTPUT_LOSS_WEIGHT = .8
        self.AUX_OUTPUT_LOSS_WEIGHT = .2
        self.ART_FRACTION = 0  # with 1, all is artificial, with 0 all is natural, in between values give a mix
        self.USE_LA_AUX_LOSS = False
        self.USE_NORMALIZATION = True
        self.USE_LA_INPUT = False
        self.VAL_LOSS_SMOOTH_WINDOW_MODEL_SELECTION = 50
        self.START_CH = 64
        self.SE2_N_THETA = 8
        self.LOAD_MODEL = True  # Continue training with model file that already exists for model name

        # Offline augmentation
        self.AUGMENT_ONLINE = False
        self.USE_READ_FILE_FOR_LAP = False
        self.NR_AUG = 10

        if self.USE_LA_INPUT and self.AUGMENT_ONLINE:
            raise Exception('USE_LA_INPUT with AUGMENT_ONLINE is not yet implemented')

        if self.USE_LA_INPUT and self.ART_FRACTION > 0:
            raise Exception('USE_LA_INPUT with artificial data is not yet implemented')

        if self.USE_LA_INPUT and self.GROUND_TRUTH == 'left_atrium':
            raise Exception('Should not be using USE_LA_INPUT with GROUND_TRUTH == \'left_atrium\'')

        # Testing and validation procedure
        self.RESIZE_BEFORE_TRAIN = False  # False or 2D tuple which represents image size in x- and y-direction
        self.USE_POST_PROCESSING = True
        self.SAVE_METRICS = True
        self.VALTEST_SET = self.TESTING_SET  # VALIDATION_SET OR TESTING_SET
        self.VALTEST_MODEL_NAMES = [self.MODEL_NAME]
        self.VALTEST_AUG_NR = 0  # Number of augmentations per image in PREDICT_SET
        self.VOXEL_OVERLAP = (0, 32, 32)
        self.BIN_THRESH = .5  # Threshold to binarize the probability images
        self.METRICS = ['Dice', 'accuracy', 'sensitivity', 'specificity', 'precision', 'TP', 'FP', 'TN', 'FN',
                        'volume_diff']

        self.CALC_PROBS = True  # If True, the probability images will be calculated with the predict function of Keras
        # and results will be saved to the disk. If False, the probability images will be loaded from disk. An error
        # will occur if these images do not exist on the disk.
        self.CALC_PROB_THRESH = True
        self.USE_ANNO_FOR_TEST = False  # Iff True, metrics will be calculated and manual annotations will be saved in
                                        # same folder as model predictions
        self.RESIZE_BEFORE_PREDICTION = False  # EITHER False or nD tuple where n == self.NR_DIM
        self.PREDICT_AUX_OUTPUT = False  # Predict the auxiliary output during test time, instead of the main output
        self.DISCARD_LAST_SLICE = False  # Set prediction of last slice to zero

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