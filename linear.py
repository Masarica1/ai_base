from numpy import array
from numpy.random import rand
import matplotlib.pyplot as plt

from data import make_linear_data

if __name__ == '__main__':
    lr = 0.001
    n, w, b = 100, rand() * 2, rand() * 2
    data = make_linear_data(100, 0.5, 0.5)

    x_arr = array([data[i][0] for i in range(100)])
    y_arr = array([data[i][1] for i in range(100)])

    dl_dw, dl_db, loss = 0, 0, 0
    loss_list = []

    for _ in range(1000):
        dl_dw = 2 * sum(x_arr * (w * x_arr - y_arr + b)) / n
        dl_db = 2 * sum(w * x_arr - y_arr + b) / n
        loss = sum((w * x_arr + b - y_arr) ** 2) / n

        w -= lr * dl_dw
        b -= lr * dl_db
        loss_list.append(loss)

    plt.scatter(x_arr, y_arr)

    plt.plot(array([0, 10]), w * array([0, 10]) + b, color='red')
    plt.show()

    print(w, b, loss)

