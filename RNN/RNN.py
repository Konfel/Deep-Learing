# -*- coding: utf-8 -*-

import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size  # memory
        self.output_size = output_size

        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,  # [batch, sequence, feature]
        )

        for parameter in self.rnn.parameters():
            nn.init.normal_(parameter, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, data, hidden_prev):
        out, hidden_prev = self.rnn(data, hidden_prev)
        # [1, sequence, h] -> [sequence, h]
        out = out.view(-1, self.hidden_size)
        # [sequence, h] -> [sequence, 1]
        out = self.linear(out)
        # [sequence, 1] -> [1, sequence, 1]
        out = out.unsqueeze(dim=0)
        return out, hidden_prev