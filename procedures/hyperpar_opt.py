def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from bayes_opt import BayesianOptimization
from core.train import Train
from core.helper_functions import Helper
from core.settings import Settings
from core.test import Test

from contextlib import contextmanager
import sys
sys.path.append("./")

import os

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


MAIN_FOLDER = 'hp_la_1/'
h = Helper(Settings())
bo_path = h.getBOPath(MAIN_FOLDER)
nr_steps_path = h.getNrStepsPath(MAIN_FOLDER)
bo = -1


def target(learning_rate_power, dropout, loss_function):
    global bo
    if bo != -1:
        pickle.dump(bo, open(bo_path, "wb"))

    # return (1 - (learning_rate_power - .6) ** 2) * (1 - (dropout - .2) ** 2) * (1 - (art_fraction - .2) ** 2)

    domains = {
        # 'unet_depth': (3, 5),
        'learning_rate_power': (-5, -3),
        # 'patch_size_factor': (1, 6),
        'dropout': (0, 1),
        # 'art_fraction': (0, 1),
        # 'feature_map_inc_rate': (1., 2.),
        'loss_function': (0, 1)
    }

    # print(domains.keys())
    hp = {}
    for k in domains.keys():
        mx = domains[k][1]
        mn = domains[k][0]
        new_value = mn + (mx - mn) * eval(k)
        hp[k] = new_value

    print(' '.join(list(hp.keys())))
    print(' '.join([str(i) for i in list(hp.values())]))
    # return hp['unet_depth'] * hp['learning_rate_power'] * hp['patch_size_factor'] * hp['dropout'] * \
    #        hp['feature_map_inc_rate'] * -1 * hp['loss_function']

    s = Settings()

    # hp['unet_depth'] = int(round(hp['unet_depth']))
    # hp['patch_size_factor'] = int(round(hp['patch_size_factor']))

    loc = locals()
    args_name = [arg for arg in inspect.getfullargspec(target).args]

    model_nr = pickle.load(open(nr_steps_path, "rb")) + 1

    s.MODEL_NAME = MAIN_FOLDER + str(model_nr)

    s.VALTEST_MODEL_NAMES = [s.MODEL_NAME]

    s.DROPOUT = hp['dropout']
    s.LEARNING_RATE = math.pow(10, hp['learning_rate_power'])
    # s.ART_FRACTION = hp['art_fraction']

    # s.UNET_DEPTH = hp['unet_depth']
    # s.PATCH_SIZE = (1, hp['patch_size_factor'] * 64, hp['patch_size_factor'] * 64)
    # s.FEATURE_MAP_INC_RATE = hp['feature_map_inc_rate']
    s.LOSS_FUNCTION = 'dice' if hp['loss_function'] < .5 else 'weighted_binary_cross_entropy'

    # s.NR_DIM = int(round(hp['nr_dim']))
    # if s.NR_DIM == 3:
    #     s.PATCH_SIZE = (3, hp['patch_size_factor'] * 32, hp['patch_size_factor'] * 32)
    # elif s.NR_DIM == 2:
    #     s.PATCH_SIZE = (1, hp['patch_size_factor'] * 64, hp['patch_size_factor'] * 64)
    # else:
    #     raise Exception('Wrong number of dimensions: {}'.format(s.NR_DIM))

    with suppress_stdout():
        h = Helper(s)

        Train(s, h).train()

        metric_means, metric_sds = Test(s, h).test()

    pickle.dump(model_nr, open(nr_steps_path, "wb"))
    return metric_means[s.MODEL_NAME]['Dice']


def visBoResValues(r):
    # print(r)
    print(r)
    a = r['all']
    m = r['max']
    params = ['step'] + ['Value'] + list(a['params'][0].keys())
    # print(params)
    data = []
    print(a['values'][0])
    for i in range(len(a['values'])):
        data.append([i] + [a['values'][i]] + list(a['params'][i].values()))

    print(list(m['max_params'].values()))
    data.append(['MAX'] + [m['max_val']] + list(m['max_params'].values()))
    print(tabulate(data, headers=params, tablefmt='orgtbl'))


def hyperpar_opt():
    resume_previous = False
    only_inspect_bo = False

    global bo

    if only_inspect_bo:
        bo = pickle.load(open(bo_path, "rb"))
        visBoResValues(bo.res)

        return

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if resume_previous:
        bo = pickle.load(open(bo_path, "rb"))
    else:
        pickle.dump(0, open(nr_steps_path, "wb"))
        # bo = BayesianOptimization(target, {
        #     'unet_depth': (2, 4),
        #     'learning_rate_power': (-6, -1),
        #     'patch_size_factor': (1, 6),
        #     'nr_dim': (2, 2),
        #     'dropout': (0, 1),
        #     'feature_map_inc_rate': (1., 2.),
        #     'loss_function': (0, 1)
        # })
        bo = BayesianOptimization(target, {
            # 'unet_depth': (0, 1),
            'learning_rate_power': (0, 1),
            # 'patch_size_factor': (0, 1),
            'dropout': (0, .8),
            # 'art_fraction': (0, 1),
            # 'feature_map_inc_rate': (0, 1),
            'loss_function': (0, 1)
        })

        bo.maximize(init_points=10, n_iter=0)

    # bo.maximize(init_points=0, n_iter=30, acq='ei')
    bo.maximize(init_points=0, n_iter=100, acq='ucb', kappa=5)

    visBoResValues(bo.res)
    pickle.dump(bo, open(bo_path, "wb"))


if __name__ == '__main__':
    hyperpar_opt()

