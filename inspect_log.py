from helper_functions import *
from settings import *
import pickle
import matplotlib.pyplot as plt
import math


class LogInspector:
    def __init__(self, s, h):
        self.s = s
        self.h = h

    def smooth(self, x, w):
        pre = math.floor(w / 2)
        post = w - pre
        for i in range(len(x)):  # range(pre, len(x) - post):
            s = max(0, i - pre)
            e = min(len(x), i + post)
            x[i] = np.mean(x[s:e])
        return x

    def inspect_log(self):
        log_path = self.h.getLogPath(self.s.MODEL_NAME)
        log = pickle.load(open(log_path, "rb"))

        print(log)

        logs_to_output = ['stopped_early', 'lowest_val_loss', 'lowest_val_loss_i']
        for log_name in logs_to_output:
            if log_name in log:
                print('{:>22} = {}'.format(log_name, log[log_name]))
            else:
                print('{:>22} = absent in this log file'.format(log_name))

        settings_to_output = ['MODEL_NAME', 'FN_CLASS_WEIGHT', 'UNET_DEPTH', 'LEARNING_RATE', 'PATCH_SIZE', 'DROPOUT',
                              'FEATURE_MAP_INC_RATE']
        for name in settings_to_output:
            try:
                print('s.{:>20} = {}'.format(name, eval("log['settings'].{}".format(name))))
            except:
                print('s.{:>20} = absent in this log file'.format(name))

        w = 50
        orig_lw = 1
        smooth_lw = 2

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(log['training']['loss'], lw=orig_lw, alpha=.3)
        plt.plot(self.smooth(log['training']['loss'], w), lw=smooth_lw)
        plt.title('Training loss')

        plt.subplot(2, 2, 2)
        plt.plot(log['training']['accuracy'], lw=orig_lw, alpha=.3)
        plt.plot(self.smooth(log['training']['accuracy'], w), lw=smooth_lw)
        plt.title('Training accuracy')

        plt.subplot(2, 2, 3)
        plt.plot(log['validation']['loss'], lw=orig_lw, alpha=.3)
        plt.plot(self.smooth(log['validation']['loss'], w), lw=smooth_lw)
        plt.title('Validation loss')

        plt.subplot(2, 2, 4)
        plt.plot(log['validation']['accuracy'], lw=orig_lw, alpha=.3)
        plt.plot(self.smooth(log['validation']['accuracy'], w), lw=smooth_lw)
        plt.title('Validation accuracy')
        plt.show()


if __name__ == "__main__":
    s = Settings()
    h = Helper(s)
    log_inspector = LogInspector(s, h)
    log_inspector.inspect_log()
