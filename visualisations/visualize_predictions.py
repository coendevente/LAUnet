from core.helper_functions import Helper
from core.settings import Settings
import numpy as np
import SimpleITK as sitk
import vtk

from vtk.util.numpy_support import numpy_to_vtk
from IPython.display import Image


def vtk_show(renderer, width=400, height=300):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    renderWindow.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    # data = str(buffer(writer.GetResult()))
    data = writer.GetResult()

    return Image(data)

dctITKtoVTK = {sitk.sitkInt8: vtk.VTK_TYPE_INT8,
               sitk.sitkInt16: vtk.VTK_TYPE_INT16,
               sitk.sitkInt32: vtk.VTK_TYPE_INT32,
               sitk.sitkInt64: vtk.VTK_TYPE_INT64,
               sitk.sitkUInt8: vtk.VTK_TYPE_UINT8,
               sitk.sitkUInt16: vtk.VTK_TYPE_UINT16,
               sitk.sitkUInt32: vtk.VTK_TYPE_UINT32,
               sitk.sitkUInt64: vtk.VTK_TYPE_UINT64,
               sitk.sitkFloat32: vtk.VTK_TYPE_FLOAT32,
               sitk.sitkFloat64: vtk.VTK_TYPE_FLOAT64}
dctVTKtoITK = dict(zip(dctITKtoVTK.values(),
                       dctITKtoVTK.keys()))

def convertTypeITKtoVTK(typeITK):
    if typeITK in dctITKtoVTK:
        return dctITKtoVTK[typeITK]
    else:
        raise ValueError("Type not supported")

def convertTypeVTKtoITK(typeVTK):
    if typeVTK in dctVTKtoITK:
        return dctVTKtoITK[typeVTK]
    else:
        raise ValueError("Type not supported")


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
    # verts, faces, normals, values = measure.marching_cubes_lewiner(a, 0)
    # print(verts)
    # mesh = Poly3DCollection(verts[faces], alpha=0.5)
    # mesh.set_facecolor(facecolor)
    # ax.add_collection3d(mesh)

    # v_a = numpy_support.numpy_to_vtk(num_array=a.ravel(), deep=True, array_type=vtk.VTK_STRUCTURED_POINTS)
    # v_a = vtk.vtkDataObject()
    # v_a.SetInformation(v_bit_a)

    # n2vtk = vtk.vtkImageImport()  # Converter
    # n2vtk.SetArr
    #
    # contour = vtk.vtkDiscreteMarchingCubes()
    # contour.SetInputData(v_a)
    # contour.SetValue(0, .5)
    # contour.Update()

    # writer = vtk.vtkPolyDataWriter()
    # writer.SetInputData(contour)
    # writer.SetFileName('contour.vtk')
    # writer.Update()
    # mc = vtk.vtkMarchingCubes(v_a)
    # print(v_a)

    n_a = np.ravel(sitk.GetArrayFromImage(a), order='C')

    v_a = vtk.vtkImageData()
    v_a.SetSpacing(a.GetSpacing())
    v_a.SetOrigin(a.GetOrigin())
    v_a.SetDimensions(a.GetSize())
    print(a.GetPixelID())
    v_a.SetScalarType(convertTypeITKtoVTK(sitk.sitkInt8), vtk.vtkInformation())
    v_a.SetNumberOfScalarComponents(a.GetNumberOfComponentsPerPixel(), vtk.vtkInformation())

    print('a')
    v_a_to_VTK = numpy_to_vtk(n_a, deep=True, array_type=convertTypeITKtoVTK(a.GetPixelID()))
    print('b')
    v_a.GetPointData().SetScalars(v_a_to_VTK)

    fow = vtk.vtkFileOutputWindow()
    fow.SetFileName('ow.txt')

    ow = vtk.vtkOutputWindow()
    ow.SetInstance(fow)

    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(v_a)
    contour.SetValue(0, .5)
    contour.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(contour.GetOutput())
    writer.SetFileName('contour.vtk')
    writer.Update()

    # renderer = vtk.vtkRenderer()
    # renderer.SetBackground(1.0, 1.0, 1.0)
    #
    # origin = v_a.GetOrigin()
    # extent = v_a.GetExtent()
    # spacing = v_a.GetSpacing()
    # xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0]
    # yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1]
    # xd = (extent[1] - extent[0] + 1) * spacing[0]
    # yd = (extent[3] - extent[2] + 1) * spacing[1]
    # camera = renderer.GetActiveCamera()
    # d = camera.GetDistance()
    # camera.SetParallelScale(0.5 * yd)
    # camera.SetFocalPoint(xc, yc, 0.0)
    # camera.SetPosition(xc, yc, +d)
    # renderer.SetActiveCamera(camera)
    #
    # mapper = vtk.vtkImageSliceMapper()
    # mapper.SetInputData(v_a)
    # mapper.SetOrientationToX()
    # mapper.SetSliceNumber(v_a.GetDimensions()[0] // 2)
    #
    # actor = vtk.vtkImageActor()
    # actor.SetMapper(mapper)
    # actor.GetProperty().SetOpacity(0.5)
    #
    # renderer.AddActor(actor)
    #
    # display(vtk_show(renderer, 800, 800))


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

        # gt_i = sitk.GetArrayFromImage(gt_i)
        # pred_i = sitk.GetArrayFromImage(pred_i)

        # reader = vtk.vtkNIFTIImageReader()
        # reader.SetFile
        # gt_i = reader.S
        # pred_i = vtk.vtkNIFTIImageReader(pred_path)

        gt.append(gt_i)
        pred.append(pred_i)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    vis_bw(-1, gt[0], facecolor='r')
    # vis_bw(ax, pred[0], facecolor='b')
    # vis_bw(ax, gt[0] != pred[0])
    # set_lims(ax, gt[0], pred[0])

    # plt.tight_layout()
    # plt.show()