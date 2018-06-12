import vtk
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import scipy
from helper_functions import Helper
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as VN


def get_screenshot(path, view, center=False):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()

    fow = vtk.vtkFileOutputWindow()
    fow.SetFileName('ow.txt')
    ow = vtk.vtkOutputWindow()
    ow.SetInstance(fow)

    surf = reader.GetOutput()
    surf.ColorCells(1, 0, 0)

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
    dist_x = 50
    dist_z = 0

    if view == 'posterior':
        pos = (center[0] + dist_z, 0, center[2] + dist_x)
    elif view == 'anterior':
        pos = (center[0] - dist_z, 0, center[2] - dist_x)
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
    return im[120:380, :], center


if __name__ == '__main__':
    nrs = [1, 10, 11, 12, 13, 15, 20, 21, 22, 24]
    # nrs = [11]
    paths = []
    views = []
    for nr in nrs:
        paths.append('C:/Users/cdv18/Downloads/drive-download-20180612T152454Z-001/anno_image_{}_-1.nii.vtk'.format(nr))
        paths.append('C:/Users/cdv18/Downloads/drive-download-20180612T152454Z-001/prob_thresh_image_{}_-1.nii.vtk'
                     .format(nr))
        views.append('anterior')
        views.append('anterior')
        paths.append('C:/Users/cdv18/Downloads/drive-download-20180612T152454Z-001/anno_image_{}_-1.nii.vtk'.format(nr))
        paths.append('C:/Users/cdv18/Downloads/drive-download-20180612T152454Z-001/prob_thresh_image_{}_-1.nii.vtk'
                     .format(nr))
        views.append('posterior')
        views.append('posterior')
    grid_size = (len(nrs), 4)

    grid_all = []

    center = -1

    for y in range(grid_size[0]):
        grid_row = []
        for x in range(grid_size[1]):
            idx = y * grid_size[1] + x
            path = paths[idx]
            view = views[idx]

            print(idx)

            if idx % 2 == 0:
                center = False

            screenshot, center = get_screenshot(path, view, center=center)
            grid_row.append(screenshot)
        grid_all.append(grid_row)

    rows = []
    for y in range(grid_size[0]):
        rows.append(np.concatenate(grid_all[y], axis=1))

    img_out = np.concatenate(rows, axis=0)

    # sitk.WriteImage(sitk.GetImageFromArray(img_out), 'grid.png')
    scipy.misc.imsave('grid.png', img_out)

    plt.figure()
    plt.imshow(img_out)
    plt.show()
