# -*- coding:utf-8 -*-

import time
from utils import audio_util

from torch.utils.data import DataLoader
from datasets.librispeech import LibriSpeech
from datasets.st_cmds_20170001_1 import ST_CMDS_20170001_1
from datasets.voxceleb2 import VoxCeleb2
from datasets.voxceleb1 import VoxCeleb1
from datasets.merged_dataset import MergedDataset

from nets.speaker_net_cnn import SpeakerNetFC

if __name__ == '__main__':
    # root_directory = '/data/open_source_dataset/ST-CMDS-20170001_1-OS'
    root_directory = '/data/open_source_dataset/test'
    dataset_type_name = 'test'
    st_cmds_20170001_1_test = VoxCeleb1(root_directory, dataset_type_name, sr=16000, n_mels=512)

    dl = DataLoader(st_cmds_20170001_1_test, batch_size=8)

    speakers_0 = st_cmds_20170001_1_test.get_speaker_data_by_nid(0, 8, 3)
    print(speakers_0.shape)

    speakers_012 = st_cmds_20170001_1_test.get_batch_speaker_data_with_icon(3, 10)
    print(speakers_012[0].shape, speakers_012[1].shape, speakers_012[2])

    # for ii, (y_a, y_p, y_n, p_nid, p_did, n_nid, n_did) in enumerate(dl):
    #     print(ii, y_a.shape, y_p.shape, y_n.shape, p_nid.shape, p_did.shape, n_did.shape, n_did.shape)
    #
    # aa = st_cmds_20170001_1_test.identity_speaker_data_gen_forever(8)
    # for ii, speakers in enumerate(aa):
    #     print(ii, speakers.shape)

    net = SpeakerNetFC(3, 0.8)
    out = net(speakers_012[0])
    print(out.shape)
