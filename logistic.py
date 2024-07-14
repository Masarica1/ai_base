from numpy import array, e, linspace
from numpy.random import rand
import matplotlib.pyplot as plt

from data import make_logistic_data


def s(t): return 1./(1 + e**(-t))


if __name__ == '__main__':
    lr = 0.001
    n, w, b = 100, rand() * 2, rand() * 2
    data = make_logistic_data(100, 0.5)

    x = array([data[i][0] for i in range(100)])
    y = array([data[i][1] for i in range(100)])

    dl_dw, dl_db, loss = 0, 0, 0
    loss_list = []

    for _ in range(500000):
        p_y = w * x + b
        dl_dw = 2 * sum(p_y * (s(p_y) - y) * s(p_y) * (1 - s(p_y))) / n
        dl_db = 2 * sum((s(p_y) - y) * s(p_y) * (1 - s(p_y))) / n
        loss = sum((s(p_y) - y) ** 2) / n

        w -= dl_dw * lr
        b -= dl_db * lr
        loss_list.append(loss)

    x_arr = linspace(0, 1, 100)
    plt.plot(x_arr, s(w * x_arr + b), color='red')
    plt.scatter(x, y)
    plt.title('Result of Logistic Regression')

    plt.show()
    print(w, b, loss)
