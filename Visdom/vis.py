# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision

from visdom import Visdom

from time import process_time

use_gpu = torch.cuda.is_available()
# use_gpu = False
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


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # 降维: 784 -> 200 -> 10
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


global_step = 0


def run_training(train_loader, net, criteon, optimizer, visdom):
    if use_gpu:
        net = net.to(device_gpu)
        criteon = criteon.to(device_gpu)

    global global_step
    for batch, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        if use_gpu:
            data = data.to(device_gpu)
            target = target.to(device_gpu)

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Visdom 可视化
        global_step += 1
        visdom.line([loss.item()], [global_step],
                    win="training", update="append")  # loss 随 global_step 的变化曲线

    return loss.item()


def run_testing(test_loader, net, criteon, visdom):
    if use_gpu:
        net = net.to(device_gpu)
        criteon = criteon.to(device_gpu)

    global global_step

    testing_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        image = data
        data = data.view(-1, 28 * 28)
        if use_gpu:
            data = data.to(device_gpu)
            target = target.to(device_gpu)

        logits = net(data)
        testing_loss += criteon(logits, target).item()

        prediction = logits.data.max(1)[1]
        correct += prediction.eq(target.data).sum()
        # 或
        # prediction = logits.argmax(dim=1)
        # correct += prediction.eq(target).float().sum().item()

    n = len(test_loader.dataset)
    average_loss = testing_loss / n
    accuracy = correct / n

    # Visdom 可视化
    if use_gpu:
        accuracy = accuracy.to(device_cpu)
    visdom.line([[testing_loss, accuracy]], [global_step],
                win="testing", update="append")  # testing_loss 和 accuracy 随 global_step 的变化曲线
    # visdom.image(image[0, :, :, :],
    #              win="one image", opts=dict(title="one image"))  # 当前 batch 的第一张图片
    # visdom.images(data.view(-1, 1, 28, 28),
    #               nrow=4, win="dataset", opts=dict(title="dataset"))  # 该 batch 的所有图片
    # visdom.text(str(prediction.detach().numpy()),
    #             win="prediction", opts=dict(title="prediction"))  # 该 batch 的所有预测值

    return average_loss, accuracy


def work():
    batch_size = 256
    learning_rate = 0.01
    epoch_number = 10

    train_loader, test_loader = load_dataset(batch_size)

    net = MLP()
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # Visdom 可视化
    visdom = Visdom()
    visdom.line([0.], [0.], win="training", opts=dict(title="training loss"))
    visdom.line([[0.0, 0.0]], [0.], win="testing", opts=dict(title="testing loss & accuracy",
                                                       legend=["loss", "accuracy"]))

    for epoch in range(epoch_number):
        training_loss= run_training(train_loader, net, criteon, optimizer, visdom)
        print("[Training] Epoch {}, Loss: {}"
              .format(epoch, training_loss))

        average_loss, accuracy = run_testing(test_loader, net, criteon, visdom)
        print("[Testing] Average loss: {}, Accuracy = {:.2f}%"
              .format(average_loss, 100 * accuracy))

        print()

    # visdom.close()


if __name__ == "__main__":
    start_time = process_time()  # 开始时间

    work()

    finish_time = process_time()  # 结束时间
    print("Time costs ", finish_time - start_time)