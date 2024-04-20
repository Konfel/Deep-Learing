# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision

from util import plot_curve, plot_image, one_hot

from time import process_time

use_gpu = torch.cuda.is_available()
# use_gpu = False
device_cpu = torch.device("cpu:0")
device_gpu = torch.device("cuda:0") if use_gpu else device_cpu


# 加载数据集
def load_dataset():
    batch_size = 512

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       # 原始数据在 [0, 1] 范围内, 将其正则化
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# 网络类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 三层线性函数, 降维: 28 * 28 -> 256 -> 64 -> 10
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, image):  # image: [b, 1, 28, 28]
        # H1 = ReLU(X * W1 + b1)
        image = F.relu(self.fc1(image))
        # H2 = ReLU(H1 * W2 + b2)
        image = F.relu(self.fc2(image))
        # H3 = ReLU(H2 * W3 + b3)
        image = self.fc3(image)
        return image


# 训练
def train(epoch_number, train_loader, net, optimizer):
    if use_gpu:
        net = net.to(device_gpu)

    train_loss = []

    for epoch in range(epoch_number):
        for batch_idx, (image, label) in enumerate(train_loader):
            # 将 image flat 为一行: [b, 1, 28, 28] -> [b, feature]
            image = image.view(image.size(0), 28*28)
            if use_gpu:
                image = image.to(device_gpu)

            # [b, feature] -> [b, 10]
            out = net(image)
            if use_gpu:
                out = out.to(device_cpu)

            # 计算 loss
            loss = F.mse_loss(out, one_hot(label))
            train_loss.append(loss.item())

            # 每 10 个 batch 输出当前的 loss
            if batch_idx % 10 == 0:
                print(epoch, batch_idx, loss.item())

            # 反向传播梯度
            optimizer.zero_grad()  # 清空梯度
            loss.backward()
            optimizer.step()

    return train_loss


if __name__ == '__main__':
    start_time = process_time()  # 训练开始时间

    # 加载数据集
    train_loader, test_loader = load_dataset()
    image, label = next(iter(train_loader))

    # 打印数据维度, 展示数据集中的图片
    # print(image.shape, label.shape, image.min(), image.max())
    # plot_image(image, label, 'Sample')

    # 训练
    net = Net()
    # 优化的参数: [w1, b1, w2, b2, w3, b3]
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    epoch_number = 3
    train_loss = train(epoch_number, train_loader, net, optimizer)

    # 输出训练时间
    finish_time = process_time()  # 训练结束时间
    print("Time costs ", finish_time - start_time)

    # 可视化 loss 的下降
    plot_curve(train_loss)

    # 计算准确率
    total_correct = 0
    for image, label in test_loader:
        image = image.view(image.size(0), 28*28)
        if use_gpu:
            image = image.to(device_gpu)

        out = net(image)
        if use_gpu:
            out = out.to(device_cpu)

        # out: [b, 10] -> prediction: [b]
        prediction = out.argmax(dim=1)  # 最大值所在的维度编号
        correct = prediction.eq(label).sum().float().item()
        total_correct += correct
    accuracy = total_correct / len(test_loader.dataset)
    print("Test accuracy: ", accuracy)

    # 可视化一个 batch 的预测结果
    image, label = next(iter(test_loader))
    if use_gpu:
        image = image.to(device_gpu)

    out = net(image.view(image.size(0), 28*28))
    if use_gpu:
        image = image.to(device_cpu)
        out = out.to(device_cpu)

    prediction = out.argmax(dim=1)
    plot_image(image, prediction, "Test")