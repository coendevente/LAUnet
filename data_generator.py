from helper_functions import Helper
from settings import Settings
import numpy as np
from imshow_3D import imshow3D
import SimpleITK as sitk


class Adjustment:
    def apply(self, image):
        raise Exception('Adjustment.apply needs to be implemented by subclass')


class ScarFibrosis(Adjustment):
    __nb_groups = 0

    def __init__(self, nb_groups):
        self.__nb_groups = nb_groups

    def apply(self, el):
        image_out = el.get_generated_image()
        return image_out


class Wall(Adjustment):
    __wall_thickness = 0

    def __init__(self, wall_thickness):
        self.__wall_thickness = wall_thickness

    def apply(self, el):
        image_out = el.get_generated_image()
        mri_image = el.get_mri_image()

        image_out = sitk.GetArrayFromImage(sitk.BinaryDilate(sitk.GetImageFromArray(image_out), self.__wall_thickness))
        image_out = image_out - el.get_mask_image()

        return image_out


class Element:
    def __init__(self):
        self.__mask_image = np.array([])
        self.__mri_image = np.array([])
        self.__generated_image = np.array([])
        self.__adjustments = []

    def get_mask_image(self):
        return self.__mask_image

    def set_mask_image(self, image):
        self.__mask_image = image
        imshow3D(image)

    def get_mri_image(self):
        return self.__mri_image

    def set_mri_image(self, image):
        self.__mri_image = image

    def get_generated_image(self):
        return self.__generated_image

    def set_generated_image(self, image):
        self.__generated_image = image

    def init_generation(self):
        raise Exception('Element.init_generation should be implemented by subclass')

    def add_adjustment(self, adj):
        self.__adjustments.append(adj)

    def apply_adjustments(self):
        for adj in self.__adjustments:
            g = adj.apply(self)
            self.set_generated_image(g)


class LeftAtrium(Element):
    def init_generation(self):
        m = self.get_mask_image()
        self.set_generated_image(m)


class DataGenerator:
    __elements = []
    __result = np.array([])

    def generate(self):
        result = self.__elements[0].get_generated_image()
        for el in self.__elements[1:]:
            result += el.get_image()

        self.__result = result

    def add_element(self, el):
        if len(self.__elements) > 0:
            if el.get_image().shape is not self.__elements[0].get_image().shape:
                raise Exception('Image shapes of all elements should be equal. Found {}, while expected {}.'
                                .format(self.__elements[0].get_image().shape, el.get_image().shape))

        self.__elements.append(el)

    def get_result(self):
        return self.__result

    def show_result(self):
        print(self.get_result().shape)
        imshow3D(self.get_result())


if __name__ == '__main__':
    s = Settings()
    h = Helper(s)

    # Get a mask which represents a left atrium
    _, _, la_path = h.getImagePaths([24], True)
    la_image = h.loadImages(["../data/4chamber/GlassHeartNonIsotropicCT_seg.gipl"])
    imshow3D(la_image[0])

    # Make left atrium object with mask image
    la = LeftAtrium()
    la.set_mask_image((la_image[0] == 2).astype(int))
    la.init_generation()

    # Add scar and fibrosis to the image
    la.add_adjustment(Wall(2))
    la.add_adjustment(ScarFibrosis(2))
    la.apply_adjustments()

    dg = DataGenerator()
    dg.add_element(la)
    dg.generate()
    dg.show_result()
