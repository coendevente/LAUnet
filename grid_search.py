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

import itertools



@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


MAIN_FOLDER = 'la_2018_challenge_1/'
h = Helper(Settings())
bo_path = h.getBOPath(MAIN_FOLDER)
nr_steps_path = h.getNrStepsPath(MAIN_FOLDER)


def target(param_names, param_values):
    model_nr = pickle.load(open(nr_steps_path, "rb")) + 1

    s = Settings()

    param_values = list(param_values)

    for i in range(len(param_names)):
        eval('s.{}'.format(param_names[i]))
        if isinstance(param_values[i], str):
            param_values[i] = '\'' + param_values[i] + '\''
        expr = 's.{} = {}'.format(param_names[i], param_values[i])
        exec(expr)

    # print('s.DROPOUT == {}'.format(s.DROPOUT))
    # print('s.LEARNING_RATE == {}'.format(s.LEARNING_RATE))
    # print('s.LOSS_FUNCTION == {}'.format(s.LOSS_FUNCTION))

    s.MODEL_NAME = MAIN_FOLDER + str(model_nr)
    # s.VALTEST_MODEL_NAMES = [s.MODEL_NAME]
    # s.ART_FRACTION = art_fraction
    h = Helper(s)

    with suppress_stdout():
        # print('here1')
        not_model_nrs = [2]
        if model_nr not in not_model_nrs:
            # print('here2')
            Train(s, h).train()
            metric_means, metric_sds = Test(s, h).test()
        else:
            # print('here3')
            s.CALC_PROBS = False
            metric_means, metric_sds = Test(s, h).test()
            s.CALC_PROBS = True

    pickle.dump(model_nr, open(nr_steps_path, "wb"))

    return metric_means[s.MODEL_NAME]['Dice']


def get_table_row(values):
    out = ''
    for v in values:
        if out != '':
            out += ' | '
        out += '{:>15}'.format(v)
    return out


def hyperpar_opt():
    resume_previous = False
    only_inspect_bo = False

    if only_inspect_bo:
        bo = pickle.load(open(bo_path, "rb"))

        print(bo)

        return

    params = {
        'LEARNING_RATE': [1e-3, 1e-4, 1e-5],
        'DROPOUT': [0, .3, .6],
        'UNET_DEPTH': [4, 5]
    }

    param_names = sorted(params.keys())
    param_values = []
    for pname in param_names:
        param_values.append(params[pname])

    print(param_names)
    print(param_values)

    param_permutations = list(itertools.product(*param_values))
    print(param_permutations)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if resume_previous:
        bo = pickle.load(open(bo_path, "rb"))
    else:
        pickle.dump(0, open(nr_steps_path, "wb"))
        bo = {}

    header_row = get_table_row(['DICE'] + param_names)
    print(header_row)
    print('-' * len(header_row))

    for pperm in param_permutations:
        finished = False
        first_retry = True
        while not finished:
            try:
                if not first_retry:
                    print('Retrying')
                val = target(param_names, pperm)
                finished = True
            except Exception:
                if first_retry:
                    print('Failed')
                first_retry = False
                # pickle.dump(pickle.load(open(nr_steps_path, "rb")) - 1, open(nr_steps_path, "wb"))
        bo[pperm] = val

        print(get_table_row([round(val, 3)] + list(pperm)))

        pickle.dump(bo, open(bo_path, "wb"))


if __name__ == '__main__':
    hyperpar_opt()

