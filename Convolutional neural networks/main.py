# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim

from LeNet5 import LeNet5
from ResNet import ResNet

from time import process_time

use_gpu = torch.cuda.is_available()
# use_gpu = False
device_cpu = torch.device("cpu:0")
device_gpu = torch.device("cuda:0") if use_gpu else device_cpu

# 加载数据集
def load_dataset(batch_size):
    training_dataset = datasets.CIFAR10('../cifar_data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize((32, 32)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                       ]))
    testing_dataset = datasets.CIFAR10('../cifar_data', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize((32, 32)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])
                                      ]))

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)

    return training_loader, testing_loader


def run_training(training_loader, net, criteon, optimizer):
    if use_gpu:
        net = net.to(device_gpu)
        criteon = criteon.to(device_gpu)

    net.train()
    for batch, (data, target) in enumerate(training_loader):
        if use_gpu:
            data = data.to(device_gpu)
            target = target.to(device_gpu)

        # [b, 3, 32, 32] -> [b, 10]
        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def run_testing(testing_loader, net, criteon):
    if use_gpu:
        net = net.to(device_gpu)
        criteon = criteon.to(device_gpu)

    net.eval()
    testing_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testing_loader:
            if use_gpu:
                data = data.to(device_gpu)
                target = target.to(device_gpu)

            logits = net(data)
            testing_loss += criteon(logits, target).item()

            prediction = logits.argmax(dim=1)
            correct += prediction.eq(target.data).sum()

    n = len(testing_loader.dataset)
    average_loss = testing_loss / n
    accuracy = correct / n
    return average_loss, accuracy


def work():
    batch_size = 256
    learning_rate = 1e-3
    epoch_number = 10

    training_loader, testing_loader = load_dataset(batch_size)

    net = LeNet5()
    # net = ResNet()
    print(net)
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epoch_number):
        training_loss = run_training(training_loader, net, criteon, optimizer)
        print("[Training] Epoch {}, Loss: {}"
              .format(epoch, training_loss))

        net.eval()  # validate 时仍要使用层与层间的所有连接
        average_loss, accuracy = run_testing(testing_loader, net, criteon)
        print("[Validation] Average loss: {}, Accuracy = {:.2f}%"
              .format(average_loss, 100 * accuracy))

        print()


if __name__ == "__main__":
    start_time = process_time()  # 开始时间

    work()

    finish_time = process_time()  # 结束时间
    print()
    print("Time costs ", finish_time - start_time)