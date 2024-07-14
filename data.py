from numpy.random import rand, choice
import matplotlib.pyplot as plt


def make_linear_data(n: int, w: float, b: float):
    data = []

    for _ in range(n):
        x = rand() * 10
        y = w * x + b + rand() * choice([-0.1, 0.1])
        data.append([x, y])

    return data


def make_logistic_data(n: int, change: float):
    data = []

    for _ in range(n):
        x = rand()

        if x <= change:
            data.append([x, 0])
        else:
            data.append([x, 1])

    return data


if __name__ == '__main__':
    d = make_logistic_data(100, 0.5)
    plt.scatter([x[0] for x in d], [x[1] for x in d])
    plt.title("Data Set for Logistic Regression")

    plt.show()
