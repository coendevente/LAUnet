import math
import numpy as np


class Settings:
    def __init__(self):
        # Model to train
        self.GROUND_TRUTH = 'scar_fibrosis'  # 'left_atrium' / 'scar_fibrosis'
        self.PRE_OR_POST_NAME = 'post'  # 'post' / 'pre'
        self.PRE_OR_POST_XX = 'b'  # 'a' / 'b'
        # self.MODEL_NAME = 'art_scar_with_blur'
        # self.MODEL_NAME = 'art_frac_1_blur_everything'
        self.MODEL_NAME = 'gs_art_fraction/5'
        # self.MODEL_NAME = 'ps_512_lr_1e4'
        # MODEL_NAME = 'union_annotations_no_aux'
        # MODEL_NAME = 'union_annotations_with_aux'

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
        self.TRAINING_SET = [10, 19, 30, 13, 6, 8, 17, 1, 23, 22, 4, 7, 26, 5]  # 18 left out because of orientation
        self.VALIDATION_SET = [25, 24, 16, 2, 14, 28, 21]
        self.TESTING_SET = [20, 29, 11, 15, 27, 9, 3, 12]

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
        self.AUTO_CLASS_WEIGHT_N = 0  # number of samples to use for the calculation of FN_CLASS_WEIGHT if it is set to
        # 'auto'
        self.EARLY_STOPPING = True
        self.PATIENTCE_ES = 2000  # Patience of early stopping
        self.DROPOUT_AT_EVERY_LEVEL = False
        self.DROPOUT = 0
        self.FEATURE_MAP_INC_RATE = 2.
        self.LOSS_FUNCTION = 'dice'  # 'weighted_binary_cross_entropy' OR 'dice'
        self.MAIN_OUTPUT_LOSS_WEIGHT = .8
        self.AUX_OUTPUT_LOSS_WEIGHT = .2
        self.ART_FRACTION = 1  # with 1, all is artificial, with 0 all is natural, in between values give a mix
        self.USE_ANY_SCAR_AUX = False
        self.USE_NORMALIZATION = True

        # Offline augmentation
        self.AUGMENT_ONLINE = False
        self.NR_AUG = 1

        # Testing and validation procedure
        self.SAVE_METRICS = True
        self.VALTEST_SET = self.VALIDATION_SET  # OR TESTING_SET
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

        self.FLIP_PROB = .5

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

        # Scar applier
        self.PATH_TO_NO_SCAR_POST = '../data/input/post/'
        self.PATH_TO_NO_SCAR_PRE = '../data/input/pre/'
        self.NO_SCAR_NRS_PRE = range(1, 31)
        self.NO_SCAR_NRS_PRE = \
            [x for x in self.NO_SCAR_NRS_PRE if x not in [3, 5, 7, 9, 11, 12, 19, 20, 23, 25, 26, 28]]
        self.NO_SCAR_NRS_POST = range(1, 31)  # 18, 16, 2
        self.NO_SCAR_NRS_POST = [x for x in self.NO_SCAR_NRS_POST if x not in [7, 17, 23, 26, 21, 3, 12, 14, 28, 5, 18]]

        self.NR_ART = 1
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
