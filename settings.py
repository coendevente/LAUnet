import math
import numpy as np

class Settings:
    # Model to train
    GROUND_TRUTH = 'scar_fibrosis'  # 'left_atrium' / 'scar_fibrosis'
    PRE_OR_POST_NAME = 'post'  # 'post' / 'pre'
    PRE_OR_POST_XX = 'b'  # 'a' / 'b'
    MODEL_NAME = 'aux_test2'

    # Path to folders
    PATH_TO_DATA = '../data/'
    PATH_TO_RESULTS = '../results/'
    PATH_TO_MODELS = '../results/models/'
    PATH_TO_AUG = '../data/augmentations/'
    PATH_TO_ART = '../data/augmentations/artificial/'

    # Division of datasets
    ALL_NATURAL_SET = range(1, 31)
    TRAINING_SET = [10, 19, 30, 13, 6, 8, 17, 1, 23, 22, 4, 7, 26, 5]  # 18 left out because of orientation
    VALIDATION_SET = [25, 24, 16, 2, 14, 28, 21]
    TESTING_SET = [20, 29, 11, 15, 27, 9, 3, 12]

    # Patchsize
    # PATCH_SIZE = (3, 64, 64)
    # PATCH_SIZE = (1, 864, 864)
    PATCH_SIZE = (1, 384, 384)
    # PATCH_SIZE = (3, 128, 128)
    # PATCH_SIZE = (1, 400, 400)
    NR_DIM = 2  # Only 2D and 3D are supported

    # Training hyperparameters
    UNET_DEPTH = 4
    LEARNING_RATE = math.pow(10, -5)
    BATCH_SIZE = 8
    NR_BATCHES = 15000
    NR_VAL_PATCH_PER_ITER = 8
    POS_NEG_PATCH_PROP = .5  # with 1, all is positive, with 0 all is negative, in between values give a mix
    FN_CLASS_WEIGHT = 'auto'  # custom number OR 'auto'
    AUTO_CLASS_WEIGHT_N = 0  # number of samples to use for the calculation of FN_CLASS_WEIGHT if it is set to 'auto'
    EARLY_STOPPING = False
    PATIENTCE_ES = 2000  # Patience of early stopping
    DROPOUT_AT_EVERY_LEVEL = False
    DROPOUT = 0.5
    FEATURE_MAP_INC_RATE = 2.
    LOSS_FUNCTION = 'dice'  # 'weighted_binary_cross_entropy' OR 'dice'
    MAIN_OUTPUT_LOSS_WEIGHT = .8
    AUX_OUTPUT_LOSS_WEIGHT = .2
    ART_FRACTION = 0  # with 1, all is artificial, with 0 all is natural, in between values give a mix
    USE_ANY_SCAR_AUX = True

    # Offline augmentation
    AUGMENT_ONLINE = False
    NR_AUG = 100

    # Testing and validation procedure
    VALTEST_SET = VALIDATION_SET  # OR TESTING_SET
    VALTEST_MODEL_NAMES = [MODEL_NAME]
    VALTEST_AUG_NR = 0  # Number of augmentations per image in PREDICT_SET
    # VOXEL_OVERLAP = (0, 200, 200)
    VOXEL_OVERLAP = (0, 32, 32)
    BIN_THRESH = .5  # Threshold to binarize the probability images
    METRICS = ['Dice', 'accuracy', 'sensitivity', 'specificity', 'precision', 'TP', 'FP', 'TN', 'FN', 'volume_diff']

    CALC_PROBS = False  # If True, the probability images will be calculated with the predict function of Keras and
    # results will be saved to the disk. If False, the probability images will be loaded from disk. An error will occur
    # if these images do not exist on the disk.

    # Data augmentation

    FLIP_PROB = .5

    ROT_MAX = 20
    ROT_MIN = -ROT_MAX

    ZOOM_MAX = .2
    ZOOM_X_MIN = 1 - ZOOM_MAX
    ZOOM_X_MAX = 1 + ZOOM_MAX
    ZOOM_Y_MIN = 1 - ZOOM_MAX
    ZOOM_Y_MAX = 1 + ZOOM_MAX

    SHEAR_MAX = .1
    SHEAR_X_MIN = -SHEAR_MAX
    SHEAR_X_MAX = SHEAR_MAX
    SHEAR_Y_MIN = -SHEAR_MAX
    SHEAR_Y_MAX = SHEAR_MAX

    NOISE_MEAN_MIN = 0
    NOISE_MEAN_MAX = 0
    NOISE_STD_MIN = 1
    NOISE_STD_MAX = 15

    # Scar applier
    PATH_TO_NO_SCAR_POST = '../data/input/post/'
    PATH_TO_NO_SCAR_PRE = '../data/input/pre/'
    NO_SCAR_NRS = range(1, 31)  # 18, 16, 2
    NR_ART = 24
    PATH_TO_ARTIFICIAL_SCAR = '../data/artificial_scar/'
    WALL_THICKNESS_MIN_MM = 3  # mm
    WALL_THICKNESS_MAX_MM = 3  # mm
    NB_GROUPS_ODDS = [0, 0, 1, 0]
    ANGLE_MIN = 10
    ANGLE_MAX = 60
    assert abs(np.sum(NB_GROUPS_ODDS) - 1) < 0.00001
    BP_STD_FACTOR_MEAN = 5
    BP_STD_FACTOR_STD = 1
    MAX_SCALE_EDGE_MM = .75  # mm
    SF_REMOVE_DILATION_MM = 1.3  # mm
    NOISE_RESAMPLE_FACTOR_MM = 1.3  # mm


