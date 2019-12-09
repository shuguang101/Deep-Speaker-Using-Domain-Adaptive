# -*- coding:utf-8 -*-


import random
import copy
import torch
import numpy as np

from torch.utils.data import Dataset


class BasicDataset(Dataset):
    #
    dataset_name = 'basic_dataset'

    def __get_speaker_dict__(self, root_directory):
        # return a dict, which key is speaker id, value is a set() of audio path
        raise NotImplementedError

    def __init__(self, root_directory, sr, n_fft,
                 n_overlap, win_length, nframes, fixed_anchor=True,
                 dataset_list=None, do_augmentation=False):
# self.sr = sr
# self.n_fft = n_fft
# self.n_overlap = n_overlap
# self.win_length = win_length
# self.nframes = nframes
# self.fixed_anchor = fixed_anchor
# self.do_augmentation = do_augmentation
#
# self.audio_file_list = []
# self.index2id = dict()
# # self.id2indexes_set = dict()
# self.sid2num = dict()
# self.sid_list = list()
# self.dataset_list = dataset_list
# self.dataset_name = dataset_name
#
# # speaker_dict_cache_file = './caches/%s_%s_speaker_dict' % (self.__class__.__name__, dataset_name)
# # using_cache = os.path.exists(speaker_dict_cache_file)
# using_cache = False
#
# if root_directory is not None:
#     print('%d [%s] scanning audio files in' % (using_cache, dataset_name), root_directory, end=', ')
# else:
#     print('%d [%s] using %d datasets' % (using_cache, dataset_name, len(dataset_list)), end=', ')
#
# # if using_cache:
# #     with open(speaker_dict_cache_file, "rb") as f:
# #         speaker_dict = pickle.load(f)
# # else:
# #     speaker_dict = self.__get_speaker_dict__(root_directory, dataset_name)
# #     with open(speaker_dict_cache_file, "wb") as f:
# #         pickle.dump(speaker_dict, f)
# speaker_dict = self.__get_speaker_dict__(root_directory, dataset_name)
#
# speaker_dict = {k: v for k, v in speaker_dict.items() if len(v) >= 8}
# self.speaker_dict = copy.deepcopy(speaker_dict)
# self.n_speakers = len(speaker_dict)
#
# # unsort  may be result unmatched label for ident speaker_id when run secoend times
# for ii, speaker_id in enumerate(sorted(speaker_dict.keys())):
#     self.sid2num[speaker_id] = ii
#     self.sid_list.append([])
#
# while len(speaker_dict) > 0:
#     for speaker_id in list(sorted(speaker_dict.keys())):
#         path = speaker_dict[speaker_id].pop()
#
#         index = len(self.audio_file_list)
#         self.index2id[index] = speaker_id
#         self.sid_list[self.sid2num[speaker_id]].append(index)
#         # self.id2indexes_set[speaker_id] = self.id2indexes_set.get(speaker_id, set()) | {index}
#         self.audio_file_list.append(path)
#
#         if len(speaker_dict[speaker_id]) == 0:
#             speaker_dict.pop(speaker_id)
#
# print('total %d speakers, %d files.' % (self.n_speakers, len(self.audio_file_list)))
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
# def __len__(self):
#     length = len(self.audio_file_list)
#     return length
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
