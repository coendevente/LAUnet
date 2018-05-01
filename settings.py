# Path to folders
PATH_TO_DATA = '../data/'
PATH_TO_RESULTS = '../results/'
PATH_TO_MODELS = '../results/models/'

# Model to train
PRE_OR_POST_NAME = 'post'  # OR 'pre'
PRE_OR_POST_XX = 'b'  # OR 'a'
MODEL_NAME = 'test_unet'

# Division of datasets
TRAINING_SET = [10, 19, 30, 13, 6, 8, 17, 1, 23, 18, 22, 4, 7, 26, 5]
VALIDATION_SET = [24]  # [25, 24, 16, 2, 14, 28, 21]
TESTING_SET = [20, 29, 11, 15, 27, 9, 3, 12]

# Patchsize
PATCH_SIZE = (3, 64, 64)

# Training hyperparameters
BATCH_SIZE = 4
NR_BATCHES = 15000
NR_VAL_PATCH_PER_ITER = 4
POS_NEG_PATCH_PROP = 0.5
FN_CLASS_WEIGHT = 4500

# Testing and validation procedure
VALTEST_SET = VALIDATION_SET  # OR TESTING_SET
VALTEST_MODEL_NAMES = [MODEL_NAME]
VALTEST_AUG_NR = 0  # Number of augmentations per image in PREDICT_SET
VOXEL_OVERLAP = (0, 32, 32)
BIN_THRESH = .5  # Threshold to binarize the probability images
METRICS = ['Dice', 'accuracy', 'sensitivity', 'specificity', 'precision', 'TP', 'FP', 'TN', 'FN', 'volume']

CALC_PROBS = True  # If True, the probability images will be calculated with the predict function of Keras and results
# will be saved to the disk. If False, the probability images will be loaded from disk. An error will occur if these
# images do not exist on the disk.

# Data augmentation
ROT_MIN = -10
ROT_MAX = 10
ZOOM_X_MIN = .95
ZOOM_X_MAX = 1.1
ZOOM_Y_MIN = .95
ZOOM_Y_MAX = 1.1
SHEAR_X_MIN = -.04
SHEAR_X_MAX = .04
SHEAR_Y_MIN = -.04
SHEAR_Y_MAX = .04
NOISE_MEAN_MIN = 0
NOISE_MEAN_MAX = 4
NOISE_STD_MIN = 1
NOISE_STD_MAX = 5
