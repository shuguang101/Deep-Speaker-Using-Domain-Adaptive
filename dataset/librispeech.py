# -*- coding:utf-8 -*-

import os
from dataset.basic_dataset import BasicDataset


class LibriSpeech(BasicDataset):

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
                    flac_files = set(filter(lambda fname: fname.endswith('.ogg'), files))
                    flac_files = set(map(lambda fname: os.path.join(p_dir, fname), flac_files))
                    if len(flac_files) > 0:
                        speaker_dict[speaker_id] = speaker_dict.get(speaker_id, set()) | flac_files

        return speaker_dict
