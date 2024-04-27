# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision

from time import process_time

# use_gpu = torch.cuda.is_available()
use_gpu = False
device_cpu = torch.device("cpu:0")
device_gpu = torch.device("cuda:0") if use_gpu else device_cpu


# 加载数据集
def load_dataset(batch_size):
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
        batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def initialization():
    # 降维: 784 -> 200 -> 10
    w1, b1 = torch.randn(200, 784, requires_grad=True), torch.zeros(200, requires_grad=True)
    w2, b2 = torch.randn(200, 200, requires_grad=True), torch.zeros(200, requires_grad=True)
    w3, b3 = torch.randn(10, 200, requires_grad=True), torch.zeros(10, requires_grad=True)

    # 不初始化会梯度消失
    torch.nn.init.kaiming_normal_(w1)
    torch.nn.init.kaiming_normal_(w2)
    torch.nn.init.kaiming_normal_(w3)

    return w1, b1, w2, b2, w3, b3


def forward(x, parameters):
    w1, b1, w2, b2, w3, b3 = parameters
    x = x @ w1.t() + b1
    x = F.relu(x)
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)  # 该 relu() 可去掉
    return x


def run_training(train_loader, parameters, criteon, optimizer):
    for batch, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)

        logits = forward(data, parameters)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def run_testing(test_loader, parameters, criteon):
    testing_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)

        logits = forward(data, parameters)
        testing_loss += criteon(logits, target).item()

        prediction = logits.data.max(1)[1]
        correct += prediction.eq(target.data).sum()

    n = len(test_loader.dataset)
    average_loss = testing_loss / n
    accuracy = correct / n
    return average_loss, accuracy


def work():
    batch_size = 256
    learning_rate = 0.01
    epoch_number = 5

    train_loader, test_loader = load_dataset(batch_size)
    w1, b1, w2, b2, w3, b3 = initialization()

    parameters = [w1, b1, w2, b2, w3, b3]
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.SGD(parameters, lr=learning_rate)

    for epoch in range(epoch_number):
        training_loss = run_training(train_loader, parameters, criteon, optimizer)
        print("[Training] Epoch {}, Loss: {}"
              .format(epoch, training_loss))

        average_loss, accuracy = run_testing(test_loader, parameters, criteon)
        print("[Testing] Average loss: {}, Accuracy = {:.2f}%"
              .format(average_loss, 100 * accuracy))

        print()


if __name__ == "__main__":
    start_time = process_time()  # 开始时间

    work()

    finish_time = process_time()  # 结束时间
    print("Time costs ", finish_time - start_time)