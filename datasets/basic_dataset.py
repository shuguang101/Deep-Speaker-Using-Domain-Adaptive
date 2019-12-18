# -*- coding:utf-8 -*-


import random
import copy
import torch
import numpy as np

from torch.utils.data import Dataset
from utils import audio_util
from utils import pil_util


class BasicDataset(Dataset):

    def __get_speaker_dict__(self, root_directory, dataset_type_name):
        # return a dict, which key is speaker id, value is a set() of audio path
        raise NotImplementedError

    def __init__(self, root_directory, dataset_type_name='train', speaker_dict_tuple=(), **kwargs):
        """
        kwargs(other_params)列表及默认值:
            fixed_anchor: False,
            sr: 44100,
            audio_file_min_duration: 3.0,
            used_duration: 2.0,
            icon_size: 15,
            icon_alpha: 0.35,
            do_augmentation: True,
            n_fft: 2048,
            win_length: 1103,
            hop_length: 275,
            window: blackman,
            n_mels:256,
        """

        # 数据集根目录
        self.root_directory = root_directory
        # 数据集类型名称: train, dev, test等
        self.dataset_type_name = dataset_type_name
        # 类型为:[dict(),dict()], 每个dict(speaker_id -> audio_path)均代表一个数据集
        self.speaker_dict_list = list(speaker_dict_tuple)
        # 其他参数
        self.other_params = dict(kwargs)
        # 原始说话人id -> 域id(domain id)
        self.sid2did_dict = dict()
        # 域个数
        self.num_of_domain = len(speaker_dict_tuple)
        # 选取的原始音频帧长度
        self.used_nframe = kwargs.get('used_duration', 2.0) * kwargs.get('sr', 44100)

        # 获取录音文件字典
        if root_directory is not None:
            print('[%s] scanning audio files in' % dataset_type_name, root_directory, end=', ', flush=True)
        # speaker_id -> audio_path set
        speaker_dict = self.__get_speaker_dict__(root_directory, dataset_type_name)
        # 由于使用triplet_loss单个说话人的录音个数需要大于2
        speaker_dict = {k: v for k, v in speaker_dict.items() if len(v) >= 2}
        self.speaker_dict = copy.deepcopy(speaker_dict)  # 录音文件字典 speaker_id -> audio_path

        # 说话人数量
        self.num_of_speakers = len(speaker_dict.keys())
        # 排序后的原始说话人id
        self.sorted_sids_list = list(sorted(speaker_dict.keys()))
        # 原始说话人id -> 整型id
        self.sid2nid_dict = dict()
        # 整型id -> 原始说话人id
        self.nid2sid_dict = dict()

        # 如果是单个的数据集, 则该数据集都属于同一个域, 则令domain_id=0
        if len(self.sid2did_dict) == 0:
            for speaker_id in self.sorted_sids_list:
                self.sid2did_dict[speaker_id] = 0

        # 需要对原始说话人id进行排序,否则会导致每次训练时同一个说话人有不同的label(ID)
        for i, speaker_id in enumerate(self.sorted_sids_list):
            self.sid2nid_dict[speaker_id] = i
            self.nid2sid_dict[i] = speaker_id
            # 将set转为list
            self.speaker_dict[speaker_id] = list(self.speaker_dict[speaker_id])

        rand_icon = pil_util.RandIcon(self.other_params.get('icon_size', 15),
                                      self.other_params.get('icon_alpha', 0.35))
        # nid -> icon, 通过id找寻对应的图标
        self.nid2icon_dict = dict()
        # did -> icon, 通过id找寻对应的图标
        self.did2icon_dict = dict()
        for nid in self.nid2sid_dict.keys():
            self.nid2icon_dict[nid] = rand_icon.get_random_icon().transpose([2, 0, 1]) / 255.0 * 2 - 1
        for did in range(self.num_of_domain):
            self.did2icon_dict[did] = rand_icon.get_random_icon().transpose([2, 0, 1]) / 255.0 * 2 - 1

        # 循环排列不同说话人的录音文件
        self.audio_file_list = []  # audio_path 列表
        self.audio_file_list_index2sid = dict()  # audio_file_list 索引 -> speaker_id
        while len(speaker_dict) > 0:
            for speaker_id in self.sorted_sids_list:
                if speaker_dict.__contains__(speaker_id) and len(speaker_dict[speaker_id]) > 0:
                    audio_path = speaker_dict[speaker_id].pop()
                    index = len(self.audio_file_list)
                    self.audio_file_list_index2sid[index] = speaker_id
                    self.audio_file_list.append(audio_path)
                    if len(speaker_dict[speaker_id]) == 0:
                        speaker_dict.pop(speaker_id)
        print('total %d speakers, %d files.' % (self.num_of_speakers, len(self.audio_file_list)), flush=True)

    def __read_audio_tuple__(self, p_index):
        # get positive path
        p_path = self.audio_file_list[p_index]
        p_sid = self.audio_file_list_index2sid[p_index]
        p_nid = self.sid2nid_dict[p_sid]
        p_did = self.sid2did_dict[p_sid]

        # get anchor path
        p_list = self.speaker_dict[p_sid]
        if self.other_params.get('fixed_anchor', False):
            a_path = p_list[0]
        else:
            a_path = random.choice(p_list)

        # get negative path
        n_sid = random.choice(self.sorted_sids_list)
        while n_sid == p_sid:
            n_sid = random.choice(self.sorted_sids_list)
        n_list = self.speaker_dict[n_sid]
        n_path = random.choice(n_list)
        n_nid = self.sid2nid_dict[n_sid]
        n_did = self.sid2did_dict[n_sid]

        # 读取音频数据
        y_a = audio_util.load_audio_as_mono(a_path, self.other_params.get('sr', 44100))
        y_p = audio_util.load_audio_as_mono(p_path, self.other_params.get('sr', 44100))
        y_n = audio_util.load_audio_as_mono(n_path, self.other_params.get('sr', 44100))

        # 在原始音频中随机选取一段
        y_a = self.audio_data_slice(y_a)
        y_p = self.audio_data_slice(y_p)
        y_n = self.audio_data_slice(y_n)

        return y_a, y_p, y_n, p_nid, p_did, n_nid, n_did

    def is_valid_audio(self, audio_path):
        # 通过判断该音频文件能否正常读取及音频时长进行判断
        result = True
        try:
            sr = self.other_params.get('sr', 44100)
            audio_file_min_duration = self.other_params.get('audio_file_min_duration', 3.0)

            y = audio_util.load_audio_as_mono(audio_path, sr)
            duration = y.shape[0] / sr
            if duration < audio_file_min_duration:
                result = False
        except Exception:
            result = False
        return result

    def audio_data_slice(self, audio_data):
        # 从原始音频数据中截取一定长度的音频
        index = random.randint(0, audio_data.shape[0] - self.used_nframe)
        sliced_audio_data = audio_data[index:index + self.used_nframe]
        return sliced_audio_data

    def get_features(self, audio_data):
        # 获取音频采样频率
        sr = self.other_params.get('sr', 44100)
        # 是否进行augmentation, shape会发生变化(time_dim 会边长或变短)
        do_augmentation = self.other_params.get('do_augmentation', True)
        # 获取频谱相关参数
        n_fft = self.other_params.get('n_fft', 2048)
        win_length = self.other_params.get('win_length', 1103)  # 25ms (sr=44100)
        hop_length = self.other_params.get('hop_length', 275)  # 1/4 of window size
        window = self.other_params.get('window', 'blackman')
        n_mels = self.other_params.get('n_mels', 256)

        # 进行数据增强
        if do_augmentation:
            audio_data = audio_util.do_audio_augmentation(audio_data, sr)

        # 滤波去除噪音
        audio_data = audio_util.de_noise(audio_data, sr)

        # 获取fbank特征
        f_bank = audio_util.fbank(audio_data, sr,
                                  n_fft=n_fft,
                                  win_length=win_length,
                                  hop_length=hop_length,
                                  window=window,
                                  n_mels=n_mels
                                  )

        # features shape: time_dim X feature_dim
        return f_bank

    def __getitem__(self, p_index):
        y_a, y_p, y_n, p_nid, p_did, n_nid, n_did = self.__read_audio_tuple__(p_index)

        f_a = self.get_features(y_a)
        f_p = self.get_features(y_p)
        f_n = self.get_features(y_n)

        return f_a, f_p, f_n, p_nid, p_did, n_nid, n_did

    def __len__(self):
        length = len(self.audio_file_list)
        return length

    def get_speaker_data_by_nid(self, nid, max_bs=8, min_bs=4):
        # 通过nid获取一批数据, 最少min_bs条, 最多max_bs条. 如果数据不足返回none
        speaker_data = None
        # 防止nid越界
        nid = nid % self.num_of_speakers
        sid = self.nid2sid_dict[nid]

        # 获取音频采样频率
        sr = self.other_params.get('sr', 44100)
        # 获取sid对应的音频文件路径列表
        path_list = self.speaker_dict[sid]

        speaker_data_list = []
        for i, path in enumerate(path_list):
            if i >= max_bs:
                break
            y = audio_util.load_audio_as_mono(path, sr)
            y = self.audio_data_slice(y)
            f = self.get_features(y)
            speaker_data_list.append(f)

        if len(speaker_data_list) >= min_bs:
            speaker_data = torch.tensor(np.array(speaker_data_list), dtype=torch.float32)
        return speaker_data

    def identity_speaker_data_gen_forever(self, batch_size):
        nid = 0
        while True:
            data = self.get_speaker_data_by_nid(nid, batch_size, 1)
            nid = (nid + 1) % self.num_of_speakers
            yield data

    def get_batch_speaker_data_with_icon(self, num_of_speaker, batch_size):
        # 获取一批音频数据并返回数据图标, num_of_speaker选取的说话人数量(选取几种人), batch_size选取的音频数据总数
        num_of_speaker = min(self.num_of_speakers, num_of_speaker)
        # 返回结果
        speaker_data_list = []
        speaker_icon_list = []
        speaker_id_list = []

        # 获取音频采样频率
        sr = self.other_params.get('sr', 44100)

        selected_sids = random.sample(self.sorted_sids_list, num_of_speaker)
        count = 0
        while count < batch_size:
            sid = selected_sids[count % len(selected_sids)]
            # 获取nid
            nid = self.sid2nid_dict[sid]
            # 获取文件列表
            path_list = self.speaker_dict[sid]
            path = random.choice(path_list)
            # 读取音频,抽取特征
            y = audio_util.load_audio_as_mono(path, sr)
            y = self.audio_data_slice(y)
            f = self.get_features(y)

            speaker_data_list.append(f)
            speaker_icon_list.append(self.nid2icon_dict[nid])
            speaker_id_list.append(sid)
            count += 1

        return torch.tensor(speaker_data_list, dtype=torch.float32), \
               torch.tensor(speaker_icon_list, dtype=torch.float32), \
               speaker_id_list
