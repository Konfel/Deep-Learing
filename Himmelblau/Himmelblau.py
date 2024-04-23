
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import torch

from time import process_time

# use_gpu = torch.cuda.is_available()
use_gpu = False
device_cpu = torch.device("cpu:0")
device_gpu = torch.device("cuda:0") if use_gpu else device_cpu

# Himmelblau 函数
def Himmelblau(x):  # x[] = [x, y]
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


# 绘制 Himmelblau 函数的图象
def show_graph_of_function(x_min, x_max, x_step, y_min, y_max, y_step):
    x, y = np.arange(x_min, x_max, x_step), np.arange(y_min, y_max, y_step)
    X, Y = np.meshgrid(x, y)
    Z = Himmelblau([X, Y])

    figure = plt.figure("Himmelblau")
    ax = figure.add_axes(Axes3D(figure))
    surface = ax.plot_surface(X, Y, Z, cmap=plt.get_cmap("rainbow_r"))
    figure.colorbar(surface, shrink=0.5, aspect=5)

    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


# 求 Himmelblau 函数的最小值
def find_minimum(initialization, learning_rate, step_number):
    print("Initialization: x = {}"
          .format(initialization))

    x = torch.tensor(initialization, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=learning_rate)

    for step in range(step_number):
        prediction = Himmelblau(x)

        optimizer.zero_grad()
        prediction.backward()
        optimizer.step()

        if step % 4000 == 0:
            print("    Step {}: x = {}, f(x) = {}"
                  .format(step, x.tolist(), prediction.item()))

    print("Result ({} iterations): x = {}, f(x) = {}"
          .format(step_number, x.tolist(), prediction.item()))
    print()

if __name__ == "__main__":
    show_graph_of_function(-6, 6, 0.1, -6, 6, 0.1)

    start_time = process_time()  # 开始时间

    find_minimum([0.0, 0.0], 1e-3, 20000)
    find_minimum([-1.0, 0.0], 1e-3, 20000)
    find_minimum([-4.0, 0.0], 1e-3, 20000)
    find_minimum([4.0, 0.0], 1e-3, 20000)

    # 输出花费时间
    finish_time = process_time()  # 结束时间
    print("Time costs ", finish_time - start_time)