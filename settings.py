# Model to train
GROUND_TRUTH = 'left_atrium'  # 'left_atrium' / 'scar_fibrosis'
PRE_OR_POST_NAME = 'post'  # 'post' / 'pre'
PRE_OR_POST_XX = 'b'  # 'a' / 'b'
MODEL_NAME = 'la_higher_class_weight'

# Path to folders
PATH_TO_DATA = '../data/'
PATH_TO_RESULTS = '../results/'
PATH_TO_MODELS = '../results/models/'
PATH_TO_AUG = '../data/augmentations_{}/'.format(GROUND_TRUTH)

# Division of datasets
TRAINING_SET = [10, 19, 30, 13, 6, 8, 17, 1, 23, 22, 4, 7, 26, 5]  # 18 left out because of orientation
VALIDATION_SET = [10]  # [25, 24, 16, 2, 14, 28, 21]
TESTING_SET = [20, 29, 11, 15, 27, 9, 3, 12]

# Patchsize
# PATCH_SIZE = (3, 64, 64)
PATCH_SIZE = (1, 480, 480)
NR_DIM = 2  # Only 2D and 3D are supported

# Training hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NR_BATCHES = 15000
NR_VAL_PATCH_PER_ITER = 4
POS_NEG_PATCH_PROP = 0.5
FN_CLASS_WEIGHT = 500
AUGMENT_ONLINE = False

# Testing and validation procedure
VALTEST_SET = VALIDATION_SET  # OR TESTING_SET
VALTEST_MODEL_NAMES = [MODEL_NAME]
VALTEST_AUG_NR = 0  # Number of augmentations per image in PREDICT_SET
VOXEL_OVERLAP = (0, 200, 200)
BIN_THRESH = .5  # Threshold to binarize the probability images
METRICS = ['Dice', 'accuracy', 'sensitivity', 'specificity', 'precision', 'TP', 'FP', 'TN', 'FN', 'volume']

CALC_PROBS = False  # If True, the probability images will be calculated with the predict function of Keras and results
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
NOISE_MEAN_MAX = 30
NOISE_STD_MIN = 1
NOISE_STD_MAX = 15

# Offline augmentation
NR_AUG = 100