
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt

from RNN import RNN

from time import process_time

use_gpu = torch.cuda.is_available()
# use_gpu = False
device_cpu = torch.device("cpu:0")
device_gpu = torch.device("cuda:0") if use_gpu else device_cpu


# 随机在 sin(x) 图象中长度为 10 的区间内采样 points_per_step 个点
def run_sampling(points_per_step):
    start = np.random.randint(3, size=1)[0]  # 起点
    time_steps = np.linspace(start, start + 10, points_per_step)
    data = np.sin(time_steps)
    data = data.reshape(points_per_step, 1)  # 变为列
    x = torch.tensor(data[:-1]).float().view(1, points_per_step - 1, 1)
    y = torch.tensor(data[1:]).float().view(1, points_per_step - 1, 1)
    return time_steps, x, y


def run_training(x, y, net, criteon, optimizer, hidden_prev):
    if use_gpu:
        net = net.to(device_gpu)
        criteon = criteon.to(device_gpu)
        x, y = x.to(device_gpu), y.to(device_gpu)
        hidden_prev = hidden_prev.to(device_gpu)

    net.train()

    logits, hidden_prev = net(x, hidden_prev)
    hidden_prev = hidden_prev.detach()  # 声明一个指向原变量存放位置的新变量, 但不计算梯度
    loss = criteon(logits, y)

    net.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def run_testing(x, y, net, criteon, hidden_prev):
    if use_gpu:
        net = net.to(device_gpu)
        criteon = criteon.to(device_gpu)
        x, y = x.to(device_gpu), y.to(device_gpu)
        hidden_prev = hidden_prev.to(device_gpu)

    net.eval()

    data = x[:, 0, :]
    predictions = []
    for _ in range(x.shape[1]):
        data = data.view(1, 1, 1)
        prediction, hidden_prev = net(data, hidden_prev)
        data = prediction

        if use_gpu:
            prediction = prediction.to(device_cpu)
        predictions.append(prediction.detach().numpy().ravel()[0])
    return predictions


def show_visualization(time_steps, x, y, predictions):
    x = x.data.numpy().ravel()  # 变为一维数组
    y = y.data.numpy()
    # plt.scatter(time_steps[:-1], x.ravel(), s=30)  # 目标数据的散点
    plt.plot(time_steps[:-1], x.ravel())  # 目标数据的曲线
    plt.scatter(time_steps[1:], predictions, c="orange", s=30)  # 预测数据的散点

    plt.legend(["target", "prediction"], loc="upper right")
    plt.xlabel("time")
    plt.ylabel("y")
    plt.show()


def work():
    points_per_step = 50  # sequence

    input_size = 1
    hidden_size = 16
    output_size = 1
    learning_rate = 0.01
    epoch_number = 6000

    net = RNN(input_size, hidden_size, output_size)
    criteon = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    hidden_prev = torch.zeros(1, 1, hidden_size)  # h_0

    for epoch in range(epoch_number):
        time_steps, x, y = run_sampling(points_per_step)
        training_loss = run_training(x, y, net, criteon, optimizer, hidden_prev)
        if epoch % 400 == 0:
            print("[Training] Epoch {}, Loss: {}"
                  .format(epoch, training_loss))
            print()

    hidden_prev = torch.zeros(1, 1, hidden_size)  # h_0
    time_steps, x, y = run_sampling(points_per_step)
    predictions = run_testing(x, y, net, criteon, hidden_prev)
    show_visualization(time_steps, x, y, predictions)


if __name__ == '__main__':
    start_time = process_time()  # 开始时间

    work()

    finish_time = process_time()  # 结束时间
    print()
    print("Time costs ", finish_time - start_time)