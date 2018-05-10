import math

class Settings:
    # Model to train
    GROUND_TRUTH = 'scar_fibrosis'  # 'left_atrium' / 'scar_fibrosis'
    PRE_OR_POST_NAME = 'post'  # 'post' / 'pre'
    PRE_OR_POST_XX = 'b'  # 'a' / 'b'
    MODEL_NAME = 'hyperpar_opt_09_05_2/22'

    # Path to folders
    PATH_TO_DATA = '../data/'
    PATH_TO_RESULTS = '../results/'
    PATH_TO_MODELS = '../results/models/'
    PATH_TO_AUG = '../data/augmentations_{}/'.format(GROUND_TRUTH)

    # Division of datasets
    TRAINING_SET = [10, 19, 30, 13, 6, 8, 17, 1, 23, 22, 4, 7, 26, 5]  # 18 left out because of orientation
    VALIDATION_SET = [25, 24, 16, 2, 14, 28, 21]
    TESTING_SET = [20, 29, 11, 15, 27, 9, 3, 12]

    # Patchsize
    # PATCH_SIZE = (3, 64, 64)
    # PATCH_SIZE = (1, 864, 864)
    PATCH_SIZE = (1, 384, 384)
    # PATCH_SIZE = (1, 400, 400)
    NR_DIM = 2  # Only 2D and 3D are supported

    # Training hyperparameters
    UNET_DEPTH = 5
    LEARNING_RATE = math.pow(10, -5)
    BATCH_SIZE = 1
    NR_BATCHES = 15000
    NR_VAL_PATCH_PER_ITER = 7
    POS_NEG_PATCH_PROP = 0.5
    FN_CLASS_WEIGHT = 'auto'  # custom number OR 'auto'
    AUTO_CLASS_WEIGHT_N = 500  # number of samples to use for the calculation of FN_CLASS_WEIGHT if it is set to 'auto'
    EARLY_STOPPING = True
    PATIENTCE_ES = 1000  # Patience of early stopping
    DROPOUT_AT_EVERY_LEVEL = False
    DROPOUT = 0.5
    FEATURE_MAP_INC_RATE = 2.
    LOSS_FUNCTION = 'dice'  # 'weighted_binary_cross_entropy' OR 'dice'

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

    CALC_PROBS = True  # If True, the probability images will be calculated with the predict function of Keras and results
    # will be saved to the disk. If False, the probability images will be loaded from disk. An error will occur if these
    # images do not exist on the disk.

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