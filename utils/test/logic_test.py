import numpy as np
from utils.layer1_logic import choose_x1_x2
from utils.layer2_logic import compute_weights_biases_layer2_classic


def test_choose_x1_x2_lowest_activation_1():
    X = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    weight = np.array([1, 1])
    bias = 0
    radius = 3.0

    x_1, x_2 = choose_x1_x2(X, weight, bias, radius)

    expected_x1 = np.array([-1, -1])
    expected_x2 = np.array([1, 1])

    np.testing.assert_array_equal(x_1, expected_x1)
    np.testing.assert_array_equal(x_2, expected_x2)


def test_choose_x1_x2_lowest_activation_2():
    X = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [0, -0.5]])
    weight = np.array([1, 1])
    bias = 0
    radius = 2.0

    x_1, x_2 = choose_x1_x2(X, weight, bias, radius)

    expected_x1 = np.array([-1, -1])
    expected_x2 = np.array([0, -0.5])

    np.testing.assert_array_equal(x_1, expected_x1)
    np.testing.assert_array_equal(x_2, expected_x2)


def test_compute_weights_biases_layer2_classic():
    X = np.array([[1]])
    y = np.array([1])
    weights = [np.array([[1, -1]]), np.array([[1], [1]])]
    biases = [np.array([-1, -1]), np.array([0])]
    weights_l1 = np.array([[1, -1]])
    biases_l1 = np.array([0, 0])

    weights_l2, biases_l2 = compute_weights_biases_layer2_classic(
        X, y, weights, biases, weights_l1, biases_l1
    )

    expected_biases_l2 = np.array([-1])

    np.testing.assert_array_equal(weights_l2, weights[1])
    np.testing.assert_array_equal(biases_l2, expected_biases_l2)


test_choose_x1_x2_lowest_activation_1()
test_choose_x1_x2_lowest_activation_1()
test_compute_weights_biases_layer2_classic()
