from settings import Settings
from helper_functions import Helper
from train import Train
from test import Test

# print('Started 4')
# s = Settings()
# s.MODEL_NAME = 'la_2018_challenge_convpl_depth_2/4'
# s.VALTEST_MODEL_NAMES = [s.MODEL_NAME]
# # s.UNET_DEPTH = 4
# # s.NR_CONV_PER_CONV_BLOCK = 2
# # s.START_CH = 64
# h = Helper(s)
# # t = Train(s, h)
# # t.train()
# t = Test(s, h)
# t.test()
# print('Ended 4')

# del t
print('Started 5')
s = Settings()
s.MODEL_NAME = 'la_2018_challenge_convpl_depth_2/5'
s.VALTEST_MODEL_NAMES = [s.MODEL_NAME]
s.UNET_DEPTH = 5
s.NR_CONV_PER_CONV_BLOCK = 2
s.START_CH = 32
h = Helper(s)
t = Train(s, h)
t.train()
t = Test(s, h)
t.test()
print('Ended 5')

del t
print('Started 6')
s = Settings()
s.MODEL_NAME = 'la_2018_challenge_convpl_depth_2/6'
s.VALTEST_MODEL_NAMES = [s.MODEL_NAME]
s.UNET_DEPTH = 6
s.NR_CONV_PER_CONV_BLOCK = 2
s.START_CH = 32
h = Helper(s)
t = Train(s, h)
t.train()
t = Test(s, h)
t.test()
print('Ended 6')