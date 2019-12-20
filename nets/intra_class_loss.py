# -*- coding:utf-8 -*-

import torch
import torch.nn as nn


class IntraClassLoss(nn.Module):

    def __init__(self, radius):
        super(IntraClassLoss, self).__init__()
        self.radius = nn.Parameter(torch.tensor(radius * 1.0, dtype=torch.float32), requires_grad=False)

    def forward(self, positive):
        mean_val = torch.mean(positive, 0)
        distances = torch.norm(positive - mean_val, dim=1)
        loss = torch.clamp(torch.mean(distances) - self.radius, min=0.0)

        return loss
