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

from tabulate import tabulate


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


MAIN_FOLDER = 'gs_art_fraction_2/'
h = Helper(Settings())
bo_path = h.getBOPath(MAIN_FOLDER)
nr_steps_path = h.getNrStepsPath(MAIN_FOLDER)


def target(art_fraction):
    model_nr = pickle.load(open(nr_steps_path, "rb")) + 1

    s = Settings()
    s.MODEL_NAME = MAIN_FOLDER + str(model_nr)
    s.VALTEST_MODEL_NAMES = [s.MODEL_NAME]
    s.ART_FRACTION = art_fraction
    h = Helper(s)

    with suppress_stdout():
        not_model_nrs = []
        if model_nr not in not_model_nrs:
            Train(s, h).train()
            metric_means, metric_sds = Test(s, h).test()
        else:
            s.CALC_PROBS = False
            metric_means, metric_sds = Test(s, h).test()
            s.CALC_PROBS = True

    pickle.dump(model_nr, open(nr_steps_path, "wb"))

    return metric_means[s.MODEL_NAME]['Dice']


def hyperpar_opt():
    resume_previous = False
    only_inspect_bo = False

    if only_inspect_bo:
        bo = pickle.load(open(bo_path, "rb"))

        print(bo)

        return

    art_fractions = np.linspace(0, 1, 5)

    print('{:>20} | {:>20}'.format('art_fraction', 'value'))
    print('-'*21+'+'+'-'*21)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if resume_previous:
        bo = pickle.load(open(bo_path, "rb"))
    else:
        pickle.dump(0, open(nr_steps_path, "wb"))
        bo = {}

    for a in art_fractions:
        val = target(a)

        bo[a] = val

        print('{:>20} | {:>20}'.format(a, val))

        pickle.dump(bo, open(bo_path, "wb"))





if __name__ == '__main__':
    hyperpar_opt()

