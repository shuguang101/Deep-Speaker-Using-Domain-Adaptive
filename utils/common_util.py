# -*- coding:utf-8 -*-

import inspect

from datasets.librispeech import LibriSpeech
from datasets.st_cmds_20170001_1 import ST_CMDS_20170001_1
from datasets.voxceleb2 import VoxCeleb2
from datasets.voxceleb1 import VoxCeleb1
from datasets.merged_dataset import MergedDataset
from torch.utils.data import DataLoader


def get_all_attribute(obj):
    # 获取对象的熟悉
    all_attrs = dir(obj)
    filtered_attrs = []

    for attr in all_attrs:
        obj_attr = getattr(obj, attr)
        if not attr.startswith('__') \
                and not callable(obj_attr) \
                and not inspect.isbuiltin(obj_attr):
            filtered_attrs.append(attr)

    return filtered_attrs


def load_data(opt, **kwargs):
    # 读取数据
    st_cmds_20170001_1 = ST_CMDS_20170001_1(opt.st_cmds_20170001_1, **kwargs)
    librispeech = LibriSpeech(opt.libriSpeech, **kwargs)
    voxceleb1 = VoxCeleb1(opt.voxceleb1, **kwargs)
    voxceleb2 = VoxCeleb2(opt.voxceleb2, **kwargs)
    merged_data = MergedDataset(None, dataset_tuple=(st_cmds_20170001_1,
                                                          librispeech,
                                                          voxceleb1,
                                                          voxceleb2),
                                **kwargs)
    merged_data_loader = DataLoader(merged_data,
                                    shuffle=opt.shuffle,
                                    batch_size=opt.batch_size,
                                    num_workers=opt.num_workers,
                                    pin_memory=opt.pin_memory,
                                    timeout=opt.dataloader_timeout)

    return merged_data, merged_data_loader
