# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F


class ClippedReLU(nn.Module):

    def __init__(self, inplace=False, max_val=20.0):
        super(ClippedReLU, self).__init__()
        self.relu_fun = nn.ReLU(inplace=inplace)
        self.max_val = max_val

    def forward(self, x):
        x = self.relu_fun(x)
        x = torch.clamp(x, min=0.0, max=self.max_val)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            ClippedReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual

        out = F.relu(out, True)
        out = torch.clamp(out, min=0.0, max=20.0)

        return out


class ConvBlock(nn.Module):

    def __init__(self):
        # baidu:      [3 3 3 3]
        # voxceleb2:  [3 4 6 3]

        super(ConvBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            ClippedReLU(True),

            nn.MaxPool2d(3, stride=2, padding=1),

            # ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),

            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            ClippedReLU(True),
            # ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            ClippedReLU(True),
            # ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            ClippedReLU(True),
            # ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x


class SpeakerNetFC(nn.Module):

    def __init__(self, num_speaker, device, keep_prop=0.8):
        super(SpeakerNetFC, self).__init__()

        drop_prop = 1 - keep_prop
        self.num_speaker = num_speaker
        self.device = device

        self.conv_layer = nn.Sequential(ConvBlock())

        self.conv_fc1 = None
        self.feature_map = None
        self.feature_map_detach = None

        self.fc2 = nn.Sequential(
            nn.Dropout(p=drop_prop),
            nn.Linear(512, self.num_speaker),
            ClippedReLU(True)
        )

    def forward(self, x):
        # input shape: [batch, audio_time_dim, audio_feature_dim]
        # out shape:   [batch, 1, audio_time_dim, audio_feature_dim]
        x = torch.unsqueeze(x, 1)

        # conv_layer input: [batch, channel=1, height, width], treat the spectrum as an one channel image
        # conv_layer out: [batch, channel, audio_time_dim, audio_feature_dim]
        x = self.conv_layer(x)

        # 动态创建网络结构
        if self.conv_fc1 is None:
            self.conv_fc1 = nn.Sequential(
                nn.Conv2d(512, 512, (1, x.shape[3]), 1, 0, bias=False),
                nn.BatchNorm2d(512),
                ClippedReLU(True),
            )
            self.conv_fc1.to(self.device)

        # conv_fc1 out: [batch, channel, audio_time_dim, audio_feature_dim=1]
        x = self.conv_fc1(x)

        # avg_pool2d out: [batch, channel, audio_time_dim=1, audio_feature_dim=1]
        x = F.avg_pool2d(x, (x.shape[2], x.shape[3]))

        # x out: [batch, channel]
        x = x.view(x.size(0), -1)

        # 抽象特征
        self.feature_map = x
        self.feature_map_detach = x.detach()

        # input shape: [batch, channel], output shape: [batch, num_speaker]
        x = self.fc2(x)

        return x

    def set_dropout_keep_prop(self, keep_prop):
        drop_prop = 1 - keep_prop
        for layer in self.fc2:
            if isinstance(layer, nn.Dropout):
                layer.p = drop_prop


class SpeakerNetEM(nn.Module):

    def __init__(self, device, keep_prop=0.8):
        super(SpeakerNetEM, self).__init__()

        drop_prop = 1 - keep_prop
        self.conv_layer = nn.Sequential(ConvBlock())
        self.device = device

        self.conv_fc1 = None

        self.embedding_layer = nn.Sequential(
            nn.Dropout(p=drop_prop),
            nn.Linear(512, 512),
            ClippedReLU(True)
        )

    def forward(self, x):
        # input shape: [batch, audio_time_dim, audio_feature_dim]
        # out shape:   [batch, 1, audio_time_dim, audio_feature_dim]
        x = torch.unsqueeze(x, 1)

        # conv_layer input: [batch, channel=1, height, width], treat the spectrum as an one channel image
        # conv_layer out: [batch, channel, audio_time_dim, audio_feature_dim]
        x = self.conv_layer(x)

        # 动态创建网络结构
        if self.conv_fc1 is None:
            self.conv_fc1 = nn.Sequential(
                nn.Conv2d(512, 512, (1, x.shape[3]), 1, 0, bias=False),
                nn.BatchNorm2d(512),
                ClippedReLU(True),
            )
            self.conv_fc1.to(self.device)

        # conv_fc1 out: [batch, channel, audio_time_dim, audio_feature_dim=1]
        x = self.conv_fc1(x)

        # avg_pool2d out: [batch, channel, audio_time_dim=1, audio_feature_dim=1]
        x = F.avg_pool2d(x, (x.shape[2], x.shape[3]))

        # x out: [batch, channel]
        x = x.view(x.size(0), -1)

        # input shape: [batch, channel], output shape: [batch, channel]
        x = self.embedding_layer(x)

        x = F.normalize(x, p=2, dim=1)

        return x

    def set_dropout_keep_prop(self, keep_prop):
        drop_prop = 1 - keep_prop
        for layer in self.embedding_layer:
            if isinstance(layer, nn.Dropout):
                layer.p = drop_prop
