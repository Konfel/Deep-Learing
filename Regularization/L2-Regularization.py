# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torchvision import datasets, transforms

from time import process_time

use_gpu = torch.cuda.is_available()
# use_gpu = False
device_cpu = torch.device("cpu:0")
device_gpu = torch.device("cuda:0") if use_gpu else device_cpu


# 加载数据集
def load_dataset(batch_size):
    training_dataset = datasets.MNIST('../../mnist_data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))
    testing_dataset = datasets.MNIST('../../mnist_data', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ]))

    testing_loader = torch.utils.data.DataLoader(testing_dataset,
                                               batch_size=batch_size, shuffle=True)

    training_dataset, validation_dataset = torch.utils.data.random_split(training_dataset, [50000, 10000])
    training_loader = torch.utils.data.DataLoader(training_dataset,
                                              batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                              batch_size=batch_size, shuffle=True)

    return training_loader, testing_loader, validation_loader


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


def run_training(training_loader, net, criteon, optimizer):
    if use_gpu:
        net = net.to(device_gpu)
        criteon = criteon.to(device_gpu)

    for batch, (data, target) in enumerate(training_loader):
        data = data.view(-1, 28 * 28)
        if use_gpu:
            data = data.to(device_gpu)
            target = target.to(device_gpu)

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def run_validation(validation_loader, net, criteon):
    if use_gpu:
        net = net.to(device_gpu)
        criteon = criteon.to(device_gpu)

    validation_loss = 0
    correct = 0
    for data, target in validation_loader:
        data = data.view(-1, 28 * 28)
        if use_gpu:
            data = data.to(device_gpu)
            target = target.to(device_gpu)

        logits = net(data)
        validation_loss += criteon(logits, target).item()

        prediction = logits.data.max(1)[1]
        correct += prediction.eq(target.data).sum()
        # 或
        # prediction = logits.argmax(dim=1)
        # correct += prediction.eq(target).float().sum().item()

    n = len(validation_loader.dataset)
    average_loss = validation_loss / n
    accuracy = correct / n
    return average_loss, accuracy


def run_testing(testing_loader, net, criteon):
    if use_gpu:
        net = net.to(device_gpu)
        criteon = criteon.to(device_gpu)

    testing_loss = 0
    correct = 0
    for data, target in testing_loader:
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

    n = len(testing_loader.dataset)
    average_loss = testing_loss / n
    accuracy = correct / n
    return average_loss, accuracy


def work():
    batch_size = 256
    learning_rate = 0.01
    epoch_number = 10

    training_loader, testing_loader, validation_loader = load_dataset(batch_size)
    print("[Set Split] Training Set: {}, Testing Set: {}, Validation Set: {}"
          .format(len(training_loader.dataset), len(testing_loader.dataset), len(validation_loader.dataset)))
    print()

    net = MLP()
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.01)  # L2-Regularization

    #optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    for epoch in range(epoch_number):
        training_loss = run_training(training_loader, net, criteon, optimizer)
        print("[Training] Epoch {}, Loss: {}"
              .format(epoch, training_loss))

        average_loss, accuracy = run_validation(validation_loader, net, criteon)
        print("[Validation] Average loss: {}, Accuracy = {:.2f}%"
              .format(average_loss, 100 * accuracy))

        print()

    average_loss, accuracy = run_testing(testing_loader, net, criteon)
    print("[Testing] Average loss: {}, Accuracy = {:.2f}%"
          .format(average_loss, 100 * accuracy))


if __name__ == "__main__":
    start_time = process_time()  # 开始时间

    work()

    finish_time = process_time()  # 结束时间
    print()
    print("Time costs ", finish_time - start_time)
