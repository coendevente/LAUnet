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
VALIDATION_SET = [25, 24, 16, 2, 14, 28, 21]
TESTING_SET = [20, 29, 11, 15, 27, 9, 3, 12]

# Patchsize
PATCH_SIZE = (3, 64, 64)

# Training hyperparameters
BATCH_SIZE = 2
NR_BATCHES = 2
NR_VAL_PATCH_PER_ITER = 10
POS_NEG_PATCH_PROP = 0.5
FN_CLASS_WEIGHT = 4500

# Testing and validation procedure
VALTEST_SET = VALIDATION_SET  # OR TESTING_SET
VALTEST_MODEL_NAMES = ['test_unet', 'unet_depth_4']
VALTEST_AUG_NR = 0  # Number of augmentations per image in PREDICT_SET
VOXEL_OVERLAP = (2, 32, 32)
# IMAGES_TO_BE_SHOWN = [0, 5]  # Will be shown to the user
BIN_THRESH = .5  # Threshold to binarize the probability images
METRICS = ['Dice', 'accuracy', 'sensitivity', 'specificity', 'precision']

CALC_PROBS = True