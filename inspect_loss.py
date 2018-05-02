from helper_functions import *
from settings import *
import pickle
import matplotlib.pyplot as plt


def main():
    loss_path = getLossPath(MODEL_NAME)
    loss = pickle.load(open(loss_path, "rb"))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(loss['training']['loss'])
    plt.title('Training loss')

    plt.subplot(2, 2, 2)
    plt.plot(loss['training']['accuracy'])
    plt.title('Training accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(loss['validation']['loss'])
    plt.title('Validation loss')

    plt.subplot(2, 2, 4)
    plt.plot(loss['validation']['accuracy'])
    plt.title('Validation accuracy')
    plt.show()


if __name__ == "__main__":
    main()