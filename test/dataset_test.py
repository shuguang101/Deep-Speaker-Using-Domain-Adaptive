# -*- coding:utf-8 -*-

import time
from utils import audio_util

from torch.utils.data import DataLoader
from datasets.librispeech import LibriSpeech
from datasets.st_cmds_20170001_1 import ST_CMDS_20170001_1
from datasets.voxceleb2 import VoxCeleb2
from datasets.voxceleb1 import VoxCeleb1
from datasets.merged_dataset import MergedDataset

if __name__ == '__main__':
    root_directory = '/data/open_source_dataset/ST-CMDS-20170001_1-OS'
    dataset_type_name = 'train'
    st_cmds_20170001_1_test = ST_CMDS_20170001_1(root_directory, dataset_type_name)

    dl = DataLoader(st_cmds_20170001_1_test, batch_size=8)

    for ii, (y_a, y_p, y_n, p_nid, p_did, n_nid, n_did) in enumerate(dl):
        print(ii, y_a.shape, y_p.shape, y_n.shape, p_nid.shape, p_did.shape, n_did.shape, n_did.shape)
