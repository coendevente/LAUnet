from helper_functions import Helper
from settings import Settings
import numpy as np
import SimpleITK as sitk
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.cm as cm
from mayavi import mlab
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from skimage.draw import ellipsoid
from matplotlib.colors import LightSource

import gr3


def set_lims(ax, a, b):
    points = np.argwhere(a > 0)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    points = np.argwhere(b > 0)
    x = np.concatenate((x, points[:, 0]))
    y = np.concatenate((y, points[:, 1]))
    z = np.concatenate((z, points[:, 2]))

    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(np.min(z), np.max(z))

def vis_bw(ax, a, facecolor='r'):

    verts, faces, normals, values = measure.marching_cubes_lewiner(a, 0)
    print(verts)
    mesh = Poly3DCollection(verts[faces], alpha=0.5)
    mesh.set_facecolor(facecolor)

    ax.add_collection3d(mesh)


if __name__ == '__main__':
    s = Settings()
    h = Helper(s)

    predict_path = h.getModelPredictPath(s.MODEL_NAME)

    gt = []
    pred = []

    for i in s.VALTEST_SET[:1]:
        gt_path = '{}anno_image_{}_{}.nii.gz'.format(predict_path, i, -1)
        pred_path = '{}prob_thresh_image_{}_{}.nii.gz'.format(predict_path, i, -1)

        gt_i = sitk.ReadImage(gt_path)
        pred_i = sitk.ReadImage(pred_path)

        gt_i = sitk.GetArrayFromImage(gt_i)
        pred_i = sitk.GetArrayFromImage(pred_i)

        gt.append(gt_i)
        pred.append(pred_i)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    vis_bw(gt[0], ax, facecolor='r')
    vis_bw(pred[0], ax, facecolor='b')
    # vis_bw(ax, gt[0] != pred[0])
    set_lims(ax, gt[0], pred[0])

    plt.tight_layout()
    plt.show()