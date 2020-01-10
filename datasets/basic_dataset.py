# -*- coding:utf-8 -*-


import os
import random
import copy
import glob
import torch
import pickle
import json
import datetime
import numpy as np
from collections import Iterable

from torch.utils.data import Dataset
from utils import audio_util
from utils import pil_util


class BasicDataset(Dataset):

    @staticmethod
    def get_ext(path):
        # 返回文件扩展名, 例如: .py .ogg .wav
        try:
            ext = os.path.splitext(path)[-1]
        except Exception:
            ext = None
        return ext

    @staticmethod
    def load_self(class_name):
        try:
            fdir = os.path.split(os.path.abspath(__file__))[0]
            save_dir = os.path.join(fdir, '../net_data/datasets_caches/')
            file_list = glob.glob(save_dir + '%s_*.pkl' % class_name)
            # os.path.getctime -> float, 1573207494.7295158
            file_list = list(map(lambda fp: (fp, os.path.getctime(fp)), file_list))
            # Descending
            file_list = list(sorted(file_list, key=lambda x: x[1], reverse=True))
            with open(file_list[0][0], 'rb') as f:
                self_obj = pickle.load(f)
            print('successful loaded the cache file: %s' % file_list[0][0], flush=True)
        except Exception:
            self_obj = None
        return self_obj

    def dump_self(self):
        fdir = os.path.split(os.path.abspath(__file__))[0]
        save_dir = os.path.join(fdir, '../net_data/datasets_caches/')
        class_name = self.__class__.__name__

        date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        data_file = '%s__%s.pkl' % (class_name, date_str)
        text_file = '%s__%s.txt' % (class_name, date_str)

        params = {**{'root_directory': self.root_directory,
                     'dataset_type_name': self.dataset_type_name,
                     'dataset_tuple_list': self.dataset_tuple_list},
                  **self.other_params}

        with open(os.path.join(save_dir, data_file), 'wb') as f:
            pickle.dump(self, f)
        with open(os.path.join(save_dir, text_file), 'w') as f:
            json.dump(params, f)

    def is_in_exts(self, path, ext_tuples):
        # 判断文件扩展名是否在ext_tuples内
        ext = self.get_ext(path)
        return ext in ext_tuples

    def __get_speaker_dict__(self, root_directory, dataset_type_name):
        # return a dict, which key is speaker id, value is a set() of audio path
        raise NotImplementedError

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)

        if len(args) > 0 and len(kwargs) > 0:
            is_params_changed = False
            obj_cache = cls.load_self(cls.__name__)

            if obj_cache is not None:
                if args[0] != obj_cache.root_directory:
                    is_params_changed = True

                obj_cache_kwargs = {**{'dataset_type_name': obj_cache.dataset_type_name,
                                       'dataset_tuple': obj_cache.dataset_tuple_list},
                                    **obj_cache.other_params}

                for key in obj_cache_kwargs.keys():
                    val1 = obj_cache_kwargs.get(key, list())
                    val2 = kwargs.get(key, list())
                    val1 = list(val1) if isinstance(val1, Iterable) else val1
                    val2 = list(val2) if isinstance(val2, Iterable) else val2
                    if val1 != val2:
                        is_params_changed = True
                if not is_params_changed:
                    obj = obj_cache
                    obj.is_loaded_from_pkl = True

            if obj_cache is None:
                print('read the cache file failed, re scan the audio files', flush=True)
            if is_params_changed:
                print('parameters changed, re scan the audio files', flush=True)
        return obj

    def __init__(self, root_directory, dataset_type_name='train', dataset_tuple=(), **kwargs):

        if hasattr(self, 'is_loaded_from_pkl'):
            print('[cached][%s] using audio files in' % dataset_type_name, root_directory, end=', ', flush=True)
            print('using %d speakers, %d files. #files_split_for_eval=%d ' % (self.num_of_speakers,
                                                                              len(self.audio_file_list),
                                                                              len(self.eval_used_dict) * 2), flush=True)
            return

        # 其他参数
        self.other_params = dict()
        # kwargs(other_params)列表及默认值
        self.all_params_set = {
            'fixed_anchor': False, 'sr': 16000, 'audio_file_min_duration': 3.0,
            'used_duration': 2.0, 'icon_size': 15, 'icon_alpha': 0.35, 'do_augmentation': True,
            'n_fft': 2048, 'win_length': 1103, 'hop_length': 275,
            'window': 'blackman', 'n_mels': 60, 'used_delta_orders': (1, 2)}
        for key in self.all_params_set:
            val = kwargs.get(key, None)
            if val is None:
                val = self.all_params_set[key]
                print('warning: lack the parameter "%s", using default value: %s' % (key, val))
            self.other_params[key] = val

        # 数据集根目录
        self.root_directory = root_directory
        # 数据集类型名称: train, dev, test等
        self.dataset_type_name = dataset_type_name
        # 类型为:[dict(),dict()], 每个dict(speaker_id -> audio_path)均代表一个数据集
        self.dataset_tuple_list = list(dataset_tuple)
        # 原始说话人id -> 域id(domain id)
        self.sid2did_dict = dict()
        # 域个数
        self.num_of_domain = max(1, len(dataset_tuple))
        # 选取的原始音频帧长度
        self.used_nframe = int(kwargs.get('used_duration', 2.0) * kwargs.get('sr', 44100))

        # 获取录音文件字典
        # if root_directory is not None:
        print('[%s] scanning audio files in' % dataset_type_name, root_directory, end=', ', flush=True)
        # speaker_id -> audio_path set
        speaker_dict = self.__get_speaker_dict__(root_directory, dataset_type_name)
        # 由于使用triplet_loss单个说话人的录音个数需要大于2
        speaker_dict = {k: v for k, v in speaker_dict.items() if len(v) >= 5}

        # 分离一部分数据用做speaker identification验证数据
        self.eval_used_dict = dict()
        for key in speaker_dict.keys():
            self.eval_used_dict[key] = [speaker_dict[key].pop(), speaker_dict[key].pop()]

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

        rand_icon = pil_util.RandIcon(self.other_params['icon_size'],
                                      self.other_params['icon_alpha'])
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
        print('using %d speakers, %d files. #files_split_for_eval=%d ' % (self.num_of_speakers,
                                                                          len(self.audio_file_list),
                                                                          len(self.eval_used_dict) * 2), flush=True)
        if not hasattr(self, 'is_loaded_from_pkl'):
            self.dump_self()

    def __read_audio_tuple__(self, p_index):
        # get positive path
        p_path = self.audio_file_list[p_index]
        p_sid = self.audio_file_list_index2sid[p_index]
        p_nid = self.sid2nid_dict[p_sid]
        p_did = self.sid2did_dict[p_sid]

        # get anchor path
        p_list = self.speaker_dict[p_sid]
        if self.other_params['fixed_anchor']:
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
        y_a = audio_util.load_audio_as_mono(a_path, self.other_params['sr'])
        y_p = audio_util.load_audio_as_mono(p_path, self.other_params['sr'])
        y_n = audio_util.load_audio_as_mono(n_path, self.other_params['sr'])

        return y_a, y_p, y_n, p_nid, p_did, n_nid, n_did

    def is_valid_audio(self, audio_path):
        # 通过判断该音频文件能否正常读取及音频时长进行判断
        result = True
        try:
            sr = self.other_params['sr']
            audio_file_min_duration = self.other_params['audio_file_min_duration']

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
        sr = self.other_params['sr']
        # 是否进行augmentation, shape会发生变化(time_dim 会边长或变短)
        do_augmentation = self.other_params['do_augmentation']
        # 获取频谱相关参数
        n_fft = self.other_params['n_fft']
        win_length = self.other_params['win_length']  # 25ms (sr=44100)
        hop_length = self.other_params['hop_length']  # 1/4 of window size
        window = self.other_params['window']
        n_mels = self.other_params['n_mels']
        # 是否使用动态特征, None或空元组则不使用动态特征, 如下代码默认使用1阶动态特征
        used_delta_orders = self.other_params['used_delta_orders']

        # 进行数据增强
        if do_augmentation:
            audio_data = audio_util.do_audio_augmentation(audio_data, sr)

        # 滤波去除噪音
        audio_data = audio_util.de_noise(audio_data, sr)

        # 在原始音频中随机选取一段
        audio_data = self.audio_data_slice(audio_data)

        # 获取fbank特征
        f_bank = audio_util.fbank(audio_data, sr,
                                  n_fft=n_fft,
                                  win_length=win_length,
                                  hop_length=hop_length,
                                  window=window,
                                  n_mels=n_mels
                                  )

        # features dim 标准化
        f_bank = (f_bank - np.mean(f_bank, axis=1, keepdims=True)) / (np.std(f_bank, axis=1, keepdims=True) + 2e-12)

        # 添加动态特征
        if used_delta_orders is not None \
                and isinstance(used_delta_orders, tuple) \
                and len(used_delta_orders) > 0:
            f_bank = audio_util.add_deltas_librosa(f_bank, orders=used_delta_orders)

        # features shape: [time_dim, feature_dim]
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
        sr = self.other_params['sr']
        # 获取sid对应的音频文件路径列表
        path_list = self.speaker_dict[sid]

        speaker_data_list = []
        for i, path in enumerate(path_list):
            if i >= max_bs:
                break
            y = audio_util.load_audio_as_mono(path, sr)
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
        sr = self.other_params['sr']

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
            f = self.get_features(y)

            speaker_data_list.append(f)
            speaker_icon_list.append(self.nid2icon_dict[nid])
            speaker_id_list.append(sid)
            count += 1

        return torch.tensor(speaker_data_list, dtype=torch.float32), \
               torch.tensor(speaker_icon_list, dtype=torch.float32), \
               speaker_id_list

    def gen_speaker_identification_eval_used_data(self, batch_size):
        # 使用eval_used_dict中的数据进行eval
        data_list, nid_list = [], []
        for sid in self.eval_used_dict.keys():
            nid = self.sid2nid_dict[sid]
            paths = self.eval_used_dict[sid]
            for path in paths:
                y = audio_util.load_audio_as_mono(path, self.other_params['sr'])
                f = self.get_features(y)
                data_list.append(f)
                nid_list.append(nid)
            # 生成batch数据
            if len(data_list) >= batch_size:
                data_np, nid_np = np.array(data_list[:batch_size]), np.array(nid_list[:batch_size])
                yield torch.tensor(data_np, dtype=torch.float32), torch.tensor(nid_np)

                data_list = data_list[batch_size:]
                nid_list = nid_list[batch_size:]

        if len(data_list) > 0:
            data_np, nid_np = np.array(data_list), np.array(nid_list)
            yield torch.tensor(data_np, dtype=torch.float32), torch.tensor(nid_np)
