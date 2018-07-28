from core.settings import Settings
from core.helper_functions import Helper
from core.test import Test
from core.train import Train

print('Started 1')
s = Settings()
s.MODEL_NAME = 'sf_set52_vanilla_valafter25steps_run4'
s.MODEL_NAME = '-'
s.VALTEST_MODEL_NAMES = [s.MODEL_NAME]
s.USE_LA_AUX_LOSS = False
s.USE_LA_INPUT = False
h = Helper(s)
t = Train(s, h)
t.train()
t = Test(s, h)
t.test()
print('Ended 1')

print('Started 2')
s = Settings()
s.MODEL_NAME = 'sf_set52_lainput_valafter25steps_run4'
s.VALTEST_MODEL_NAMES = [s.MODEL_NAME]
s.USE_LA_AUX_LOSS = False
s.USE_LA_INPUT = True
s.VALIDATION_SET = s.VALIDATION_SET[11:]
s.VALTEST_SET = s.VALIDATION_SET
h = Helper(s)
t = Train(s, h)
t.train()
t = Test(s, h)
t.test()
print('Ended 2')

print('Started 1')
s = Settings()
s.MODEL_NAME = 'sf_set52_auxloss_valafter25steps_run4'
s.VALTEST_MODEL_NAMES = [s.MODEL_NAME]
s.USE_LA_AUX_LOSS = True
s.USE_LA_INPUT = False
s.VALTEST_SET = s.TESTING_SET
s.PREDICT_AUX_OUTPUT = True
h = Helper(s)
t = Train(s, h)
t.train()
t = Test(s, h)
t.test()
print('Ended 1')
