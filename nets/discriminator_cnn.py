# -*- coding:utf-8 -*-

import torch.nn as nn


# Domain Adapation 判别器
# 判别器越好，生成器梯度消失越严重
# 在判别器中使用leakrelu激活函数，而不是RELU，防止梯度稀疏
class DANet(nn.Module):

    def __init__(self, num_features, num_domain):
        super(DANet, self).__init__()

        self.num_features = num_features
        self.num_domain = num_domain
        self.main = nn.Sequential(
            nn.Linear(self.num_features, 32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(32, self.num_domain),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.main(x)
        return x
