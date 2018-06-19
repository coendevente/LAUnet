import vtk
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import scipy
from helper_functions import Helper
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as VN
import math


def get_screenshot(path, view, center=False):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()

    fow = vtk.vtkFileOutputWindow()
    fow.SetFileName('ow.txt')
    ow = vtk.vtkOutputWindow()
    ow.SetInstance(fow)

    surf = reader.GetOutput()
    # surf.ColorCells(1, 0, 0)

    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    WIDTH = 640
    HEIGHT = 500
    renWin.SetSize(WIDTH, HEIGHT)

    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # mapper
    surfMapper = vtk.vtkPolyDataMapper()
    surfMapper.SetInputData(surf)

    # actor
    surfActor = vtk.vtkActor()
    surfActor.SetMapper(surfMapper)
    # surfActor.GetProperty().SetColor(1, 0, 0)

    if not center:
        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(surf)
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()
        center = centerOfMassFilter.GetCenter()
    print(center)

    camera = vtk.vtkCamera()

    dist_x = 0
    dist_z = 400

    if view == 'posterior':

        pos = (center[0] - dist_x, center[1] + 400, center[2] - dist_z)
    elif view == 'anterior':
        pos = (center[0] + dist_x, center[1] + 0, center[2] + dist_z)
    else:
        raise ValueError('view is not supported: {}'.format(view))
    camera.SetPosition(pos)
    camera.SetFocalPoint(center[0], center[1], center[2])

    # assign actor to the renderer
    ren.AddActor(surfActor)
    ren.SetActiveCamera(camera)
    ren.SetBackground(1, 1, 1)

    renWin.Render()

    # screenshot code:
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName("screenshots/screenshot.png")
    writer.SetInputData(w2if.GetOutput())
    writer.Write()

    # iren.Initialize()
    # renWin.Render()
    # iren.Start()

    im = sitk.GetArrayFromImage(sitk.ReadImage('screenshots/screenshot.png'))
    return im[50:430, 25:605], center


if __name__ == '__main__':
    nrs = np.array([70, 21, 95, 73, 78, 26, 38, 82, 47, 40, 66, 59, 13, 89, 71, 88, 37, 22, 84, 10, 97, 68, 65, 48,
 45])

    d = np.array([0.9006973680422561, 0.910006143806372, 0.9112106116568368, 0.8601871142224925, 0.7259960147934539, 0.7628879251385949, 0.8283370968273197, 0.914378003436467, 0.9261529957159189, 0.8196490038357094, 0.9119802222766654, 0.8743586059357887, 0.8684744959632821, 0.9153005083364386, 0.9223461250934463, 0.8978720592122654, 0.931094379993213, 0.8323133596744007, 0.8952515304300879, 0.7863649833930516, 0.6610801486199576, 0.9001830952523773, 0.905672853299284, 0.8840102160996202, 0.8836265065576483])
    nrs = np.flip(nrs[np.argsort(d)], 0)

    # nrs = nrs[:8]
    # nrs = nrs[-1:-9:-1]

    shell_paths = '../data/shells_predictions_testcase25/'

    # nrs = [11]
    # paths = []
    # views = []
    paths = [0] * 50
    views = [0] * 50
    img_nrs = [0] * 50

    i = 0
    for nr in nrs:
        # paths.append('{}anno_image_{}_-1.nii.vtk'.format(shell_paths, nr))
        # paths.append('{}prob_thresh_image_{}_-1.nii.vtk'.format(shell_paths, nr))
        # views.append('anterior')
        # views.append('anterior')
        # paths.append('{}anno_image_{}_-1.nii.vtk'.format(shell_paths, nr))
        # paths.append('{}prob_thresh_image_{}_-1.nii.vtk'.format(shell_paths, nr))
        # views.append('posterior')
        # views.append('posterior')

        idx = math.floor(i / 5) * 5 * 2 + (i % 5)
        print(idx)
        paths[idx] = '{}anno_image_{}_-1.nii.vtk'.format(shell_paths, nr)
        views[idx] = 'anterior'
        img_nrs[idx] = i

        idx = math.floor(i / 5) * 5 * 2 + 5 + (i % 5)
        print(idx)
        paths[idx] = '{}prob_thresh_image_{}_-1.nii.vtk'.format(shell_paths, nr)
        views[idx] = 'anterior'
        img_nrs[idx] = i

        i += 1
    # grid_size = (len(nrs), 2)
    grid_size = (10, 5)

    print(paths)

    grid_all = []

    centers = []

    for y in range(grid_size[0]):
        grid_row = []
        for x in range(grid_size[1]):
            idx = y * grid_size[1] + x
            path = paths[idx]
            view = views[idx]

            print(idx)

            if y % 2 == 0:
                center = False
            else:
                img_nr = int(img_nrs[idx])
                center = centers[img_nr]

            screenshot, center = get_screenshot(path, view, center=center)

            if y % 2 == 0:
                centers.append(center)

            if y % 2 == 0:
                screenshot[:, :, 0] = screenshot[:, :, 1]
                screenshot[:, :, 1] = screenshot[:, :, 2]

            lw = 10
            if y % 2 == 1 and y != 9:
                print(y)
                screenshot = np.pad(screenshot, ((0, lw), (0, 0), (0, 0)), mode='constant', constant_values=0)

            if (x + 1) % 5 != 0:
                screenshot = np.pad(screenshot, ((0, 0), (0, lw), (0, 0)), mode='constant', constant_values=0)

            grid_row.append(screenshot)
        grid_all.append(grid_row)

    rows = []
    for y in range(grid_size[0]):
        rows.append(np.concatenate(grid_all[y], axis=1))

    img_out = np.concatenate(rows, axis=0)

    # sitk.WriteImage(sitk.GetImageFromArray(img_out), 'grid.png')
    scipy.misc.imsave('grid_5x5.png', img_out)

    plt.figure()
    plt.imshow(img_out)
    plt.show()
