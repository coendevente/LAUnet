from settings import *
import random


def augment(I, J):
    rot = random.uniform(ROT_MIN, ROT_MAX)
    zoom = random.uniform(ZOOM_MIN, ZOOM_MAX)
    sheer = random.uniform(SHEER_MIN, SHEER_MAX)
    return I, J