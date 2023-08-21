import numpy as np


def solve_for_X(w, b):
    w_pseudo_inv = np.linalg.pinv(w)
    X = np.dot(b, w_pseudo_inv)
    return X


def find_center(w, b):
    X = solve_for_X(w, b)
    mean = np.mean(X)
    return mean


def find_magnitude(w):
    w = np.transpose(w)
    norms = [np.linalg.norm(vector) for vector in w]
    average = np.mean(norms)

    return average


def find_center_and_magnitude(w, b):
    return find_center(w, b), find_magnitude(w)
