# -*- coding: utf-8 -*-

import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convolution_unit = nn.Sequential(
            # 第 1 个卷积层: [b, 3, 32, 32] -> [b, 16, ...]
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 第 2 个卷积层: [b, 16, 32, 32] -> [b, 32, ...]
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.full_connected_unit = nn.Sequential(
            # 降维: 32 * 5 * 5 -> 32 -> 10
            nn.Linear(32 * 5 * 5, 32),  # input 维度输出可知
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # [b, 3, 32, 32] -> [b, 32, 5, 5]
        x = self.convolution_unit(x)
        # [b, 32, 5, 5] -> [b, 32 * 5 * 5]
        x = x.view(batch_size, -1)  # flatten
        # [b, 32 * 5 * 5] -> [b, 10]
        logits = self.full_connected_unit(x)
        return logits
