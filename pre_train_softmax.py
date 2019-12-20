# -*- coding:utf-8 -*-


if __name__ == '__main__':
    # use 'spawn' in main file's first line, to prevent deadlock occur
    import multiprocessing

    multiprocessing.set_start_method('spawn')

import fire
import signal
import torch

from config import opt
from utils import common_util
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from nets.speaker_net_cnn import SpeakerNetFC


def train(**kwargs):
    # 覆盖默认参数
    for k, v in kwargs.items():
        setattr(opt, k, v)
    # 获取参数
    opt_attrs = common_util.get_all_attribute(opt)
    params_dict = {k: getattr(opt, k) for k in opt_attrs}

    # 测试数据集参数
    dataset_train_param = {**params_dict, **{'dataset_type_name': 'train'}}
    dataset_test_param = {**params_dict, **{'dataset_type_name': 'test'}}

    # 读取测试数据
    merged_test, merged_test_data_loader = common_util.load_data(opt, **dataset_test_param)
    # 读取训练数据
    merged_train, merged_train_data_loader = common_util.load_data(opt, **dataset_train_param)

    # tensor board summary writer
    summary_writer = SummaryWriter(log_dir=opt.summary_log_dir)
    # 用于在tensor board中显示聚类效果
    em_mat, em_imgs, em_id = merged_test.get_batch_speaker_data_with_icon(10, 50)

    lr = opt.lr
    epoch = 1
    global_step = 1

    speaker_net = SpeakerNetFC(merged_train.num_of_speakers, opt.dropout_keep_prop)

    device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')
    map_location = lambda storage, loc: storage.cuda(0) if opt.gpu else lambda storage, loc: storage

    optimizer = torch.optim.SGD(speaker_net.parameters(),
                                lr=lr,
                                weight_decay=opt.weight_decay,
                                momentum=opt.momentum)


def exit_signal_handler(signal, frame):
    global exit_sign
    if not exit_sign:
        exit_sign = True
        print('please waiting, program exiting ......')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, exit_signal_handler)
    signal.signal(signal.SIGHUP, exit_signal_handler)
    signal.signal(signal.SIGINT, exit_signal_handler)

    fire.Fire()

    print('clean finished, program exited')
