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
    loss_path = getLossPath(MODEL_NAME)
    loss = pickle.load(open(loss_path, "rb"))

    w = 15
    orig_lw = 1
    smooth_lw = 2

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(loss['training']['loss'], lw=orig_lw, alpha=.3)
    plt.plot(smooth(loss['training']['loss'], w), lw=smooth_lw)
    plt.title('Training loss')

    plt.subplot(2, 2, 2)
    plt.plot(loss['training']['accuracy'], lw=orig_lw, alpha=.3)
    plt.plot(smooth(loss['training']['accuracy'], w), lw=smooth_lw)
    plt.title('Training accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(loss['validation']['loss'], lw=orig_lw, alpha=.3)
    plt.plot(smooth(loss['validation']['loss'], w), lw=smooth_lw)
    plt.title('Validation loss')

    plt.subplot(2, 2, 4)
    plt.plot(loss['validation']['accuracy'], lw=orig_lw, alpha=.3)
    plt.plot(smooth(loss['validation']['accuracy'], w), lw=smooth_lw)
    plt.title('Validation accuracy')
    plt.show()


if __name__ == "__main__":
    main()