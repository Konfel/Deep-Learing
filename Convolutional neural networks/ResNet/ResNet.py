# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F


class ResesidualBlock(nn.Module):
    def __init__(self, channel_in, channel_out, stride=1):
        super(ResesidualBlock, self).__init__()

        # 第 1 个卷积层
        self.convolution_unit_1 = nn.Conv2d(
            channel_in, channel_out, kernel_size=3, stride=stride, padding=1)  # stride 用于防止参数量成倍增加
        self.batch_norm_1 = nn.BatchNorm2d(channel_out)
        # 第 2 个卷积层
        self.convolution_unit_2 = nn.Conv2d(
            channel_out, channel_out, kernel_size=3, stride=1, padding=1)  # stride 用于防止参数量成倍增加
        self.batch_norm_2 = nn.BatchNorm2d(channel_out)

        self.extra_unit = nn.Sequential()  # channel_in 不等于 channel_out 时的修正
        if channel_in != channel_out:
            # [b, channel_in, h, w] -> [b, channel_out, h, w]
            self.extra_unit = nn.Sequential(
                nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=stride),  # 此处的 stride 同第 1 个卷积层
                nn.BatchNorm2d(channel_out)
            )

    def forward(self, x):
        out = F.relu(self.batch_norm_1(self.convolution_unit_1(x)))
        out = self.batch_norm_2(self.convolution_unit_2(out))

        # short cut
        out = self.extra_unit(x) + out
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # 预处理层
        self.convolution_unit = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )

        # channel: 64 -> 128 -> 256 -> 512, [h, w] 递减至 [1, 1]
        # 第 1 个 ResBlock : [b, 64, h, w] -> [b, 128, h, w]
        self.residual_block_1 = ResesidualBlock(64, 128, stride=2)
        # 第 2 个 ResBlock : [b, 128, h, w] -> [b, 256, h, w]
        self.residual_block_2 = ResesidualBlock(128, 256, stride=2)
        # 第 3 个 ResBlock : [b, 256, h, w] -> [b, 512, h, w]
        self.residual_block_3 = ResesidualBlock(256, 512, stride=2)
        # 第 4 个 ResBlock : [b, 512, h, w] -> [b, 512, h, w]
        self.residual_block_4 = ResesidualBlock(512, 512, stride=2)

        # [b, 512, 1, 1] -> [b, 10]
        self.output_unit = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        x = F.relu(self.convolution_unit(x))

        x = self.residual_block_1(x)
        x = self.residual_block_2(x)
        x = self.residual_block_3(x)
        x = self.residual_block_4(x)

        # [b, 512, h, w] -> [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # [b, 512, 1, 1] -> [b, 512 * 1 * 1]
        x = x.view(x.size(0), -1)
        x = self.output_unit(x)

        return x