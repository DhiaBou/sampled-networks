import numpy as np

from utils.layer1_logic import choose_x1_x2
from utils.layer2_logic import compute_weights_biases_layer2_classic


def test_choose_x1_x2_lowest_activation_1():
    X = np.array([[-0.4, -0.4],
                  [0.6, 1.4],
                  [1, 0],
                  [-0.4, 0.9]
                  ])
    weight = np.array([1, 1])
    bias = 0.3
    radius = 0

    x_1, x_2 = choose_x1_x2(X, weight, bias, radius, choose_x_2="angle")
    w1i_hat = (
            (x_2 - x_1)
            / (np.linalg.norm(x_2 - x_1)) ** 2
    )
    b1_hat = np.dot(x_1, w1i_hat)
    kk = x_1 + w1i_hat
    print(w1i_hat[0], " ", w1i_hat[1])
    print(b1_hat)
    print(kk[0], ",", kk[1])
    for x in X:
        print(np.dot(weight, x) - bias)
    for x in X:
        print(np.dot(w1i_hat, x) - b1_hat)


def test_choose_x1_x2_lowest_activation_5():
    X = np.array([[0, 0],
                  [0.0001, 1 / np.sqrt(2)],
                  [1 / 2 + 1 / np.sqrt(2), 1 / 2 + 1 / np.sqrt(2)],
                  ])
    print("\n##########")
    weight = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    bias = 0
    radius = 0
    x_1 = [0, 0]
    for x_2 in X:
        w1i_hat = (
                (x_2 - x_1)
                / (np.linalg.norm(x_2 - x_1)) ** 2
        )
        print(w1i_hat[0], " ", w1i_hat[1])
        print(np.linalg.norm(weight - w1i_hat))
        print("---")

    x_1, x_2 = choose_x1_x2(X, weight, bias, radius, choose_x_2="norm")
    w1i_hat = (
            (x_2 - x_1)
            / (np.linalg.norm(x_2 - x_1)) ** 2
    )
    b1_hat = np.dot(x_1, w1i_hat)
    kk = x_1 + w1i_hat
    print(w1i_hat[0], " ", w1i_hat[1])


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
