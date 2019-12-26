# -*- coding:utf-8 -*-


if __name__ == '__main__':
    # use 'spawn' in main file's first line, to prevent deadlock occur
    import multiprocessing

    multiprocessing.set_start_method('spawn')

import fire
import torch
import os
import datetime
import copy
import numpy as np

from torchnet import meter
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config import opt
from utils import common_util
from nets.speaker_net_cnn import SpeakerNetFC
from datasets.voxceleb1 import VoxCeleb1


def train(**kwargs):
    # 获取该文件的目录
    fdir = os.path.split(os.path.abspath(__file__))[0]


if __name__ == '__main__':
    fire.Fire()
