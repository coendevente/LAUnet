from settings import Settings
from helper_functions import Helper
from train import Train
from test import Test

print('Started 1')
s = Settings()
s.MODEL_NAME = 'sf_with_la_aux_d5_nc1_k32_do50_allimg512_no_la_aux'
s.VALTEST_MODEL_NAMES = [s.MODEL_NAME]
s.LOAD_MODEL = True
s.USE_LA_AUX_LOSS = False
h = Helper(s)
t = Train(s, h)
t.train()
t = Test(s, h)
t.test()
print('Ended 1')