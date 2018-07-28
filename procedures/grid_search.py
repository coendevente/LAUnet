def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from core.train import Train
from core.helper_functions import Helper
from core.settings import Settings
from core.test import Test

from contextlib import contextmanager
import sys
sys.path.append("./")

import os

import pickle

from joblib import Parallel, delayed


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# MAIN_FOLDER = 'la_2018_challenge_1/'
MAIN_FOLDER = 'sf_grid_search_05July2018/'
h = Helper(Settings())
bo_path = h.getBOPath(MAIN_FOLDER)
# nr_steps_path = h.getNrStepsPath(MAIN_FOLDER)


def target(param_names, param_values, model_nr):
    # model_nr = pickle.load(open(nr_steps_path, "rb")) + 1
    # pickle.dump(model_nr, open(nr_steps_path, "wb"))

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
    s.VALTEST_MODEL_NAMES = [s.MODEL_NAME]
    h = Helper(s)

    # with suppress_stdout():
    # print('here1')
    not_model_nrs = []
    if model_nr not in not_model_nrs:
        # print('here2')
        t = Train(s, h)
        t.train()
        del t
        metric_means, metric_sds = Test(s, h).test()
    else:
        # print('here3')
        s.CALC_PROBS = False
        metric_means, metric_sds = Test(s, h).test()
        s.CALC_PROBS = True

    return metric_means[s.MODEL_NAME]['Dice'], metric_sds[s.MODEL_NAME]['Dice']


def get_table_row(values):
    out = ''
    for v in values:
        if out != '':
            out += ' | '
        out += '{:>25}'.format(v)
    return out


def par_one(input):
    param_names, pperm, model_nr = input[0], input[1], input[2]

    finished = False
    first_retry = True
    while not finished:
        # try:
        if not first_retry:
            print('Retrying')
        mean, std = target(param_names, pperm, model_nr)
        finished = True
        # except Exception:
        #     if first_retry:
        #         print('Failed')
        #     first_retry = False
        #     time.sleep(30)
    val = '{} \pm {}'.format(round(mean, 4), round(std, 5))

    bo = pickle.load(open(bo_path, "rb"))
    bo[pperm] = val
    print(get_table_row([val] + list(pperm)))
    pickle.dump(bo, open(bo_path, "wb"))


def hyperpar_opt():
    resume_previous = False
    only_inspect_bo = False

    if only_inspect_bo:
        bo = pickle.load(open(bo_path, "rb"))

        print(bo)

        return

    # params = {
    #     'LEARNING_RATE': [1e-3, 1e-4, 1e-5],
    #     'DROPOUT': [0, .3, .6],
    #     'UNET_DEPTH': [4, 5]
    # }
    #
    # param_names = sorted(params.keys())
    # param_values = []
    # for pname in param_names:
    #     param_values.append(params[pname])
    #
    # print(param_names)
    # print(param_values)
    # param_permutations = list(itertools.product(*param_values))

    param_names = ['UNET_DEPTH', 'NR_CONV_PER_CONV_BLOCK', 'USE_LA_INPUT', 'USE_LA_AUX_LOSS']
    param_permutations = [(4, 1, False, False),
                          (5, 1, False, False),
                          (4, 1, True, False),
                          (5, 1, True, False),
                          (4, 1, False, True),
                          (5, 1, False, True),
                          (4, 2, False, False),
                          (5, 2, False, False),
                          (4, 2, True, False),
                          (5, 2, True, False),
                          (4, 2, False, True),
                          (5, 2, False, True)]

    print(param_names)
    print(param_permutations)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if resume_previous:
        bo = pickle.load(open(bo_path, "rb"))
    else:
        # pickle.dump(0, open(nr_steps_path, "wb"))
        bo = {}
    pickle.dump(bo, open(bo_path, "wb"))

    header_row = get_table_row(['DICE'] + param_names)
    print(header_row)
    print('-' * len(header_row))

    inputs = [[param_names, param_permutations[i], i] for i in range(len(param_permutations))]
    Parallel(n_jobs=2)(delayed(par_one)(i) for i in inputs)


if __name__ == '__main__':
    hyperpar_opt()

