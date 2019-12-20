# -*- coding:utf-8 -*-

import numpy as np
import datetime
import torch
import torch.nn as nn
from torchnet import meter
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from nets.speaker_net_cnn import SpeakerNetEM
# from nets.speaker_net_lstm import SpeakerNetLSTM
from nets.intra_class_loss import IntraClassLoss
from datasets.librispeech import LibriSpeech
from datasets.st_cmds_20170001_1 import ST_CMDS_20170001_1
from datasets.voxceleb2 import VoxCeleb2
from datasets.voxceleb1 import VoxCeleb1
from datasets.merged_dataset import MergedDataset
from config import opt
from utils import audio_util, metric_util

# from train import compute_equal_error_rate
from utils import pil_util

test_audio_path = 'audios/1_src2.wav'
model_path = 'checkpoints/cnn/27_1770000__2019-05-28_07_25_26.pth'

speaker_net = SpeakerNetEM(opt.dropout_keep_prop)

device = torch.device('cuda') if opt.gpu else torch.device('cpu')
map_location = lambda storage, loc: storage.cuda(0) if opt.gpu else lambda storage, loc: storage
speaker_net.to(device)

status_dict = torch.load(model_path, map_location)
speaker_net.load_state_dict(status_dict['net'])
speaker_net.eval()

data = audio_util.norm_magnitude_spectrum(audio_path, opt.sr, opt.n_fft,
                                          opt.n_overlap, opt.win_length)

# 14s
testdata = np.expand_dims(data[:3 * 300, :], 0)
testdata = testdata.reshape(-1, 300, 513)

tensor_teset_data = torch.tensor(testdata).to(device)

ret = speaker_net(tensor_teset_data)
ret = ret.cpu().detach().numpy()

sim_mat = np.zeros((ret.shape[0], ret.shape[0]))
for i in range(ret.shape[0]):
    for j in range(i, ret.shape[0]):
        if i == j:
            sim_mat[i, j] = 1
        else:
            sim = np.linalg.norm((ret[i, :] - ret[j, :]))
            sim_mat[i, j] = sim

np.set_printoptions(3)
print(sim_mat.astype(np.float32))

# used for test
dataset_p_test = {'dataset_name': 'test', 'sr': opt.sr, 'n_fft': opt.n_fft, 'win_length': opt.win_length,
                  'n_overlap': opt.n_overlap, 'nframes': opt.n_frame, 'fixed_anchor': opt.fixed_anchor}

st_cmds_20170001_1_test = ST_CMDS_20170001_1(opt.st_cmds_20170001_1, **dataset_p_test)
librispeech_test = LibriSpeech(opt.libriSpeech, **dataset_p_test)
voxceleb1_test = VoxCeleb1(opt.voxceleb1, **dataset_p_test)
voxceleb2_test = VoxCeleb2(opt.voxceleb2, **dataset_p_test)

merged_test = MergedDataset.dataset_merge([
    librispeech_test, st_cmds_20170001_1_test, voxceleb1_test, voxceleb2_test
])

thresholds_list = list(np.arange(0, 1, 0.0125, dtype=np.float32))
th_arr, frr_arr, far_arr, eer, th_root = metric_util.compute_equal_error_rate(speaker_net, thresholds_list,
                                                                              merged_test, device,
                                                                              max_people_to_compare=8,
                                                                              max_bs=32,
                                                                              min_bs=4)

pil_util.plot_eer(frr_arr, far_arr, eer, th_root, 'epoch=27', save_path='./27_with_vox2.png')
