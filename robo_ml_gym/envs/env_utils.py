import math
import numpy as np


def is_xy_close(arr1, arr2, a_tol: float):
    """check if x & y for arr1 & arr2 are within tolerance a_tol"""
    if abs(arr2[0] - arr1[0]) < a_tol:
        if abs(arr2[1] - arr1[1]) < a_tol:
            return True
    return False


def get_xy_dist(arr1, arr2):
    """return a 2D array of the x-y distances"""
    return ((arr2[0] - arr1[0])**2 + (arr2[1] - arr1[1])**2)**0.5


def point_dist(a: np.array, b: np.array):
    """ The absolute distance between two points in space """
    return abs(np.linalg.norm(a - b))


def vector_angle(a: np.array, b: np.array):
    return math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) * 180 / np.pi
