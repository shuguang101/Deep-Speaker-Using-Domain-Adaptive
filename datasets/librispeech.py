# -*- coding:utf-8 -*-

import os
from datasets.basic_dataset import BasicDataset


class LibriSpeech(BasicDataset):
    sid_pre = 'librispeech_'  # 保证id全局唯一, 添加前缀
    ext_tuples = ('.m4a',)

    def __get_speaker_dict__(self, root_directory, dataset_type_name):

        dir_dict = {'train': [], 'dev': [], 'test': []}
        for file in os.listdir(root_directory):
            path = os.path.join(root_directory, file)
            if os.path.isdir(path):
                if file.startswith('train-'):
                    dir_dict['train'].append(path)
                elif file.startswith('dev-'):
                    dir_dict['dev'].append(path)
                elif file.startswith('test-'):
                    dir_dict['test'].append(path)
                else:
                    pass

        speaker_dict = dict()
        dir_list = dir_dict[dataset_type_name]
        for directory in dir_list:
            for speaker_id in os.listdir(directory):
                path = os.path.join(directory, speaker_id)
                for p_dir, dirs, files in os.walk(path):
                    paths = set(map(lambda fname: os.path.join(p_dir, fname), files))
                    paths = set(filter(lambda fpath: self.is_in_exts(fpath, self.ext_tuples)
                                                     and self.is_valid_audio(fpath), paths))
                    if len(paths) > 0:
                        speaker_id_with_pre = self.sid_pre + speaker_id
                        speaker_dict[speaker_id_with_pre] = speaker_dict.get(speaker_id_with_pre, set()) | paths

        return speaker_dict
