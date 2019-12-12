# -*- coding:utf-8 -*-

import os
from datasets.basic_dataset import BasicDataset


class ST_CMDS_20170001_1(BasicDataset):

    def __get_speaker_dict__(self, root_directory, dataset_type_name):
        speaker_dict = dict()
        for file in os.listdir(root_directory):
            path = os.path.join(root_directory, file)
            if path.endswith('.ogg') and self.is_valid_audio(path):
                speaker_id = file[8:14]
                speaker_dict[speaker_id] = speaker_dict.get(speaker_id, set()) | {path}

        dev_size = min(int(len(speaker_dict) * 0.2), 10000)
        test_size = min(int(len(speaker_dict) * 0.2), 10000)
        train_size = len(speaker_dict) - dev_size - test_size

        train_dict = dict()
        dev_dict = dict()
        test_dict = dict()
        dataset_dict = {'train': train_dict, 'dev': dev_dict, 'test': test_dict}

        for i, (k, v) in enumerate(speaker_dict.items()):
            if i < train_size:
                train_dict[k] = v
            elif train_size <= i < train_size + dev_size:
                dev_dict[k] = v
            else:
                test_dict[k] = v

        return dataset_dict[dataset_type_name]
