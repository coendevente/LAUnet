import numpy as np
import matplotlib.pyplot as plt

"""
File with function imshow3D, which allows to view a 3D np array as an image.

Source: https://matplotlib.org/examples/pylab_examples/image_slices_viewer.html
"""

# Class that helps imshow3D
class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[self.ind, :, :], cmap="Greys_r",
                            vmin=np.min(self.X), vmax=np.max(self.X))
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def onkeypress(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.key == 'up' or event.key == 'right':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        # self.ax.imshow(self.X[self.ind, :, :], cmap="Greys_r")
        # print(self.ind)
        self.im.set_data(self.X[self.ind, :, :])
        self.ax.set_ylabel('slice %s, sum of voxels = %i' % (self.ind, np.sum(self.X[self.ind, :, :])))
        self.im.axes.figure.canvas.draw()


# Show 3D images
def imshow3D(im):
    """
    Show 3D numpy array as image. Image is displayed. User can scroll through slices or use arrow keys.
    :param im: 3D array
    :return: None
    """
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, im)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.onkeypress)
    plt.show()
