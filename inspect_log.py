from helper_functions import *
from settings import *
import pickle
import matplotlib.pyplot as plt
import math


def smooth(x, w):
    pre = math.floor(w / 2)
    post = w - pre
    for i in range(len(x)):  #range(pre, len(x) - post):
        s = max(0, i - pre)
        e = min(len(x), i + post)
        x[i] = np.mean(x[s:e])
    return x


def main():
    log_path = getLogPath(MODEL_NAME)
    log = pickle.load(open(log_path, "rb"))

    print(log)

    logs_to_output = ['fn_class_weight']

    for log_name in logs_to_output:
        if log_name in log:
            print('{} = {}'.format(log_name, log['fn_class_weight']))
        else:
            print('{} is absent in this log file'.format(log_name))

    w = 15
    orig_lw = 1
    smooth_lw = 2

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(log['training']['loss'], lw=orig_lw, alpha=.3)
    plt.plot(smooth(log['training']['loss'], w), lw=smooth_lw)
    plt.title('Training loss')

    plt.subplot(2, 2, 2)
    plt.plot(log['training']['accuracy'], lw=orig_lw, alpha=.3)
    plt.plot(smooth(log['training']['accuracy'], w), lw=smooth_lw)
    plt.title('Training accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(log['validation']['loss'], lw=orig_lw, alpha=.3)
    plt.plot(smooth(log['validation']['loss'], w), lw=smooth_lw)
    plt.title('Validation loss')

    plt.subplot(2, 2, 4)
    plt.plot(log['validation']['accuracy'], lw=orig_lw, alpha=.3)
    plt.plot(smooth(log['validation']['accuracy'], w), lw=smooth_lw)
    plt.title('Validation accuracy')
    plt.show()


if __name__ == "__main__":
    main()