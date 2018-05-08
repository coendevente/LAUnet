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


def target(unet_depth, learning_rate_power, patch_size_factor):
    s = Settings()

    unet_depth = int(round(unet_depth))
    patch_size_factor = int(round(patch_size_factor))

    loc = locals()
    args_name = [arg for arg in inspect.getfullargspec(target).args]
    # args_values = [loc[arg] for arg in inspect.getfullargspec(target).args]

    s.MODEL_NAME = 'hyperpar_opt_08_05_0/'

    first = True
    for name in args_name:
        if not first:
            s.MODEL_NAME += ','
        first = False

        s.MODEL_NAME += name + '=' + str(locals()[name])

    s.VALTEST_MODEL_NAMES = [s.MODEL_NAME]

    s.UNET_DEPTH = unet_depth
    s.LEARNING_RATE = math.pow(10, learning_rate_power)
    s.PATCH_SIZE = (1, patch_size_factor * 64, patch_size_factor * 64)

    with suppress_stdout():
        h = Helper(s)

        Train(s, h).train()

        metric_means, metric_sds = Test(s, h).test()

    return metric_means[s.MODEL_NAME]['Dice']


def hyperpar_opt():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    bo = BayesianOptimization(target, {
        'unet_depth': (2, 5),
        'learning_rate_power': (-6, -1),
        'patch_size_factor': (1, 6),
        # 'dimensionality': (2, 3)
    })
    bo.explore({
        'unet_depth': [2, 3, 4, 5, 4],
        'learning_rate_power': [-2, -5, -3, -6, -4],
        'patch_size_factor': [6, 5, 4, 2, 1],
        # 'dimensionality': [2, 3, 2, 3, 2]
    })
    bo.maximize(init_points=0, n_iter=20, kappa=3)

    print(bo.res['all'])
    print(bo.res['max'])

    h = Helper(Settings())
    bo_path = h.getBOPath()
    pickle.dump(bo, open(bo_path, "wb"))


if __name__ == '__main__':
    hyperpar_opt()
