def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from train import Train
from helper_functions import Helper
from settings import Settings
from test import Test

from contextlib import contextmanager
import sys
sys.path.append("./")

import os
import tensorflow as tf

import math
import pickle

import inspect


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


MAIN_FOLDER = 'hyperpar_opt_09_05_1/'
h = Helper(Settings())
bo_path = h.getBOPath(MAIN_FOLDER)
nr_steps_path = h.getNrStepsPath(MAIN_FOLDER)
bo = -1


def target(unet_depth, learning_rate_power, patch_size_factor, dropout, feature_map_inc_rate):
    global bo
    if bo != -1:
        pickle.dump(bo, open(bo_path, "wb"))
    # return unet_depth * learning_rate_power * patch_size_factor * do_every_level * dropout * feature_map_inc_rate * -1

    s = Settings()

    unet_depth = int(round(unet_depth))
    patch_size_factor = int(round(patch_size_factor))

    loc = locals()
    args_name = [arg for arg in inspect.getfullargspec(target).args]

    model_nr = pickle.load(open(nr_steps_path, "rb")) + 1
    pickle.dump(model_nr, open(nr_steps_path, "wb"))

    s.MODEL_NAME = MAIN_FOLDER + str(model_nr)

    s.VALTEST_MODEL_NAMES = [s.MODEL_NAME]

    s.UNET_DEPTH = unet_depth
    s.LEARNING_RATE = math.pow(10, learning_rate_power)
    s.PATCH_SIZE = (1, patch_size_factor * 64, patch_size_factor * 64)
    # s.DROPOUT_AT_EVERY_LEVEL = do_every_level >= .5
    s.DROPOUT = dropout
    s.FEATURE_MAP_INC_RATE = feature_map_inc_rate

    with suppress_stdout():
        h = Helper(s)

        Train(s, h).train()

        metric_means, metric_sds = Test(s, h).test()

    return metric_means[s.MODEL_NAME]['Dice']


def hyperpar_opt():
    resume_previous = False
    only_inspect_bo = False

    global bo

    if only_inspect_bo:
        bo = pickle.load(open(bo_path, "rb"))
        print(bo.res['all'])
        print(bo.res['max'])

        return

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if resume_previous:
        bo = pickle.load(open(bo_path, "rb"))
    else:
        pickle.dump(0, open(nr_steps_path, "wb"))
        bo = BayesianOptimization(target, {
            'unet_depth': (2, 5),
            'learning_rate_power': (-6, -1),
            'patch_size_factor': (1, 6),
            # 'do_every_level': (0, 1),
            'dropout': (0, 1),
            'feature_map_inc_rate': (1., 2.)
        })
        # bo.explore({
        #     'unet_depth': [2, 3, 4, 5, 4, 3],
        #     'learning_rate_power': [-2, -5, -3, -6, -4, -3],
        #     'patch_size_factor': [6, 6, 6, 4, 3, 2],
        #     'do_every_level': [0, 1, 0, 1, 0, 1],
        #     'dropout': [0., .2, .4, .6, .8, 1.],
        #     'feature_map_inc_rate': [1., 1.2, 1.4, 1.6, 1.8, 2.]
        # })
        bo.maximize(init_points=10, n_iter=0, kappa=5)

    bo.maximize(init_points=0, n_iter=30, kappa=5)

    print(bo.res['all'])
    print(bo.res['max'])
    pickle.dump(bo, open(bo_path, "wb"))


if __name__ == '__main__':
    hyperpar_opt()
