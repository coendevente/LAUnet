from core.settings import Settings
from core.helper_functions import Helper
from core.test import Test

s = Settings()
s.MODEL_NAME = 'sf_set52_auxloss_valafter25'
s.VALTEST_MODEL_NAMES = [
                         # 'sf_set52_vanilla_valafter25steps',
                         # 'sf_set52_vanilla_valafter25steps_run2',
                         # 'sf_set52_vanilla_valafter25steps_run3',
                         # 'sf_set52_vanilla_valafter25steps_run4',
                         # 'sf_set52_lainput_valafter25steps',
                         # 'sf_set52_lainput_valafter25steps_run2',
                         # 'sf_set52_lainput_valafter25steps_run3',
                         # 'sf_set52_lainput_valafter25steps_run4',
                         # 'sf_set52_auxloss_valafter25steps',
                         # 'sf_set52_auxloss_valafter25steps_run2',
                         # 'sf_set52_auxloss_valafter25steps_run3',
                         # 'sf_set52_auxloss_valafter25steps_run4',
                         'la_transfer',
                         # 'la_2018_challenge_convpl_depth_2/2'
                         ]
# s.GROUND_TRUTH = 'scar_fibrosis'
s.GROUND_TRUTH = 'left_atrium'
# s.USE_LA_AUX_LOSS = True
s.USE_LA_INPUT = False
s.PREDICT_AUX_OUTPUT = False

# s.VALTEST_SET = [45, 48]
s.CALC_PROBS = True
s.VALTEST_SET = s.TESTING_SET
s.USE_POST_PROCESSING = True
s.CALC_PROB_THRESH = True
h = Helper(s)
t = Test(s, h)
t.test()
