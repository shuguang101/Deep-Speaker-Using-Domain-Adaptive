# -*- coding:utf-8 -*-


import random
import copy
import torch
import numpy as np

from torch.utils.data import Dataset


class BasicDataset(Dataset):

    def __get_speaker_dict__(self, root_directory, dataset_type_name):
        # return a dict, which key is speaker id, value is a set() of audio path
        raise NotImplementedError

    def __init__(self, root_directory,
                 dataset_type_name='train',
                 speaker_dict_list=[],
                 other_params={}):
        # 数据集根目录
        self.root_directory = root_directory
        # 数据集类型名称: train, dev, test等
        self.dataset_type_name = dataset_type_name
        # 类型为:[dict(),dict()], 每个dict(speaker_id -> audio_path)均代表一个数据集
        self.speaker_dict_list = speaker_dict_list
        # 其他参数
        self.other_params = other_params

        # 获取录音文件字典
        if root_directory is not None:
            print('[%s] scanning audio files in' % dataset_type_name, root_directory, end=', ')
        speaker_dict = self.__get_speaker_dict__(root_directory, dataset_type_name)
        # 由于使用triplet_loss单个说话人的录音个数需要大于2
        speaker_dict = {k: v for k, v in speaker_dict.items() if len(v) >= 2}
        self.speaker_dict = copy.deepcopy(speaker_dict)  # 录音文件字典 speaker_id -> audio_path

        # 说话人数量
        self.num_of_speakers = len(speaker_dict)
        # 排序后的原始说话人id
        self.sorted_sids_list = list(sorted(speaker_dict.keys()))
        # 原始说话人id -> 整型id
        self.sid2nid_dict = dict()
        # 整型id -> 原始说话人id
        self.nid2sid_dict = dict()

        # 需要对原始说话人id进行排序,否则会导致每次训练时同一个说话人有不同的label(ID)
        for i, speaker_id in enumerate(self.sorted_sids_list):
            self.sid2nid_dict[speaker_id] = i
            self.nid2sid_dict[i] = speaker_id

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
        print('total %d speakers, %d files.' % (self.num_of_speakers, len(self.audio_file_list)))

    #
    # self.valid_tuple_data = self.__get_a_valid_tuple_data__()

    # def __get_a_valid_tuple_data__(self):
    #     num = self.nframes
    #     a_data, p_data, n_data, p_speaker_id, n_speaker_id = None, None, None, None, None
    #
    #     # find a valid a_data and p_data
    #     for speaker_id in self.speaker_dict.keys():
    #         path_set = self.speaker_dict[speaker_id]
    #         data_list = []
    #         for path in path_set:
    #             data = audio_util.norm_magnitude_spectrum(path, self.sr, self.n_fft,
    #                                                       self.n_overlap, self.win_length, self.do_augmentation)
    #             if data.shape[0] - num >= 0 and len(data_list) < 2:
    #                 data_list.append(data)
    #             if len(data_list) >= 2:
    #                 break
    #         if len(data_list) >= 2:
    #             a_data = data_list[0]
    #             p_data = data_list[1]
    #             p_speaker_id = speaker_id
    #             break
    #
    #     # find a valid n_data
    #     for speaker_id in self.speaker_dict.keys():
    #         if speaker_id != p_speaker_id:
    #             path_set = self.speaker_dict[speaker_id]
    #             for path in path_set:
    #                 data = audio_util.norm_magnitude_spectrum(path, self.sr, self.n_fft,
    #                                                           self.n_overlap, self.win_length, self.do_augmentation)
    #
    #                 if data.shape[0] - num >= 0:
    #                     n_data = data
    #                     n_speaker_id = speaker_id
    #                     break
    #             if n_data is not None or n_speaker_id is not None:
    #                 break
    #
    #     return a_data, p_data, n_data, p_speaker_id, n_speaker_id
    #
    # def __read_tuple_data__(self, p_index):
    #     num = self.nframes
    #     p_speaker_id = self.index2id[p_index]
    #     # p_set = self.id2indexes_set[p_speaker_id]
    #     # p_set.difference_update({p_index})
    #     pl_index = self.sid2num[p_speaker_id]
    #     p_list = self.sid_list[pl_index]
    #
    #     a_data = None
    #     if self.fixed_anchor:
    #         for ii in p_list:  # sorted(p_set):
    #             anchor_path = self.audio_file_list[ii]
    #             data = audio_util.norm_magnitude_spectrum(anchor_path, self.sr, self.n_fft,
    #                                                       self.n_overlap, self.win_length, False)
    #             if data.shape[0] - num >= 0:
    #                 a_data = data
    #                 break
    #     else:
    #         a_index_list = p_list  # list(p_set)
    #         a_index = random.choice(a_index_list)
    #         anchor_path = self.audio_file_list[a_index]
    #         a_data = audio_util.norm_magnitude_spectrum(anchor_path, self.sr, self.n_fft,
    #                                                     self.n_overlap, self.win_length, False)
    #
    #     # n_set = set(range(0, len(self.audio_file_list)))
    #     # n_set.difference_update(self.id2indexes_set[p_speaker_id])
    #
    #     # n_index_list = list(n_set)
    #     if pl_index == 0:
    #         nl_index = random.randint(1, len(self.sid_list) - 1)
    #     elif pl_index == len(self.sid_list) - 1:
    #         nl_index = random.randint(0, len(self.sid_list) - 2)
    #     else:
    #         if random.randint(0, 1) == 0:
    #             nl_index = random.randint(0, pl_index - 1)
    #         else:
    #             nl_index = random.randint(pl_index + 1, len(self.sid_list) - 1)
    #
    #     # n_index_list = list(range(0,self.sid2num[p_speaker_id])) + list(range(self.sid2num[p_speaker_id]))
    #     n_index = random.choice(self.sid_list[nl_index])
    #     n_speaker_id = self.index2id[n_index]
    #
    #     positive_path = self.audio_file_list[p_index]
    #     negative_path = self.audio_file_list[n_index]
    #
    #     p_data = audio_util.norm_magnitude_spectrum(positive_path, self.sr, self.n_fft,
    #                                                 self.n_overlap, self.win_length, self.do_augmentation)
    #     n_data = audio_util.norm_magnitude_spectrum(negative_path, self.sr, self.n_fft,
    #                                                 self.n_overlap, self.win_length, False)
    #
    #     if a_data is None or a_data.shape[0] - num < 0 or p_data.shape[0] - num < 0 or n_data.shape[0] - num < 0:
    #         a_data, p_data, n_data, p_speaker_id, n_speaker_id = self.valid_tuple_data
    #     if a_data is None or a_data.shape[0] - num < 0 or p_data.shape[0] - num < 0 or n_data.shape[0] - num < 0:
    #         return None
    #     else:
    #         return a_data, p_data, n_data, p_speaker_id, n_speaker_id
    #
    # def __getitem__(self, p_index):
    #     tuple_data = self.__read_tuple_data__(p_index)
    #
    #     if tuple_data is None:
    #         return None, None, None, None
    #
    #     a_data, p_data, n_data, p_speaker_id, n_speaker_id = tuple_data
    #
    #     num = self.nframes
    #     ra = random.randint(0, a_data.shape[0] - num)
    #     rp = random.randint(0, p_data.shape[0] - num)
    #     rn = random.randint(0, n_data.shape[0] - num)
    #
    #     return a_data[ra:ra + num, :], p_data[rp:rp + num, :], n_data[rn:rn + num, :], \
    #            self.sid2num[p_speaker_id], self.sid2num[n_speaker_id]
    #
    def __len__(self):
        length = len(self.audio_file_list)
        return length
#
# def get_speaker_i_data(self, sid_index, max_bs=8, min_bs=4):
#     num = self.nframes
#     indexes_list = self.sid_list[sid_index]
#
#     speaker_data_list = []
#     for ii, index in enumerate(indexes_list):
#         if ii >= max_bs:
#             break
#
#         audio_path = self.audio_file_list[index]
#         data = audio_util.norm_magnitude_spectrum(audio_path, self.sr, self.n_fft,
#                                                   self.n_overlap, self.win_length)
#
#         if data.shape[0] - num >= 0:
#             r = random.randint(0, data.shape[0] - num)
#             speaker_data_list.append(data[r:r + num, :])
#
#     if len(speaker_data_list) >= min_bs:
#         return torch.tensor(np.array(speaker_data_list), dtype=torch.float32)
#     else:
#         return None
#
# def identity_speaker_data_gen(self, batch_size):
#     num = self.nframes
#
#     for speaker_id in sorted(self.speaker_dict.keys()):
#         indexes_list = self.sid_list[self.sid2num[speaker_id]]
#         length = len(indexes_list)
#         speaker_data_list = []
#         i = 0
#         while len(speaker_data_list) < batch_size:
#             index = indexes_list[i % length]
#             audio_path = self.audio_file_list[index]
#             data = audio_util.norm_magnitude_spectrum(audio_path, self.sr, self.n_fft,
#                                                       self.n_overlap, self.win_length)
#             if data.shape[0] - num >= 0:
#                 r = random.randint(0, data.shape[0] - num)
#                 speaker_data_list.append(data[r:r + num, :])
#             if i + 1 >= length and len(speaker_data_list) <= 0:
#                 break
#             i += 1
#
#         if len(speaker_data_list) == batch_size:
#             yield torch.tensor(np.array(speaker_data_list), dtype=torch.float32)
#
# def identity_speaker_data_gen_forever(self, batch_size):
#     data_gen = self.identity_speaker_data_gen(batch_size)
#     while True:
#         data = next(data_gen, None)
#         if data is not None:
#             yield data
#         else:
#             data_gen = self.identity_speaker_data_gen(batch_size)
#             data = next(data_gen, None)
#             yield data
#
# def get_bath_speakers_with_icon(self, nspeaker, num, icon_size, icon_alpha):
#     nspeaker = min(self.n_speakers, nspeaker)
#     speaker_data_list = []
#     speaker_icon_list = []
#     speaker_id_list = []
#
#     count = 0
#     ri = RandIcon(icon_size, icon_alpha)
#     for speaker_id in sorted(self.speaker_dict.keys()):
#         indexes_list = self.sid_list[self.sid2num[speaker_id]]
#         icon = ri.get_random_icon().transpose([2, 0, 1]) / 255.0 * 2 - 1
#
#         audio_data_list = []
#         for index in indexes_list:
#             audio_path = self.audio_file_list[index]
#             data = audio_util.norm_magnitude_spectrum(audio_path, self.sr, self.n_fft,
#                                                       self.n_overlap, self.win_length)
#             if data.shape[0] > self.nframes:
#                 audio_data_list.append(data)
#
#         if len(audio_data_list) > 0:
#             for i in range(num):
#                 data = audio_data_list[i % len(audio_data_list)]
#                 r = random.randint(0, data.shape[0] - self.nframes)
#                 speaker_data_list.append(data[r:r + self.nframes, :])
#                 speaker_icon_list.append(icon)
#                 speaker_id_list.append(speaker_id)
#             count += 1
#
#         if count >= nspeaker:
#             break
#
#     return torch.tensor(speaker_data_list, dtype=torch.float32), \
#            torch.tensor(speaker_icon_list, dtype=torch.float32), \
#            speaker_id_list
