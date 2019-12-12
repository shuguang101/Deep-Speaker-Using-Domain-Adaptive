# -*- coding:utf-8 -*-

import os
from datasets.basic_dataset import BasicDataset


class VoxCeleb2(BasicDataset):
    # speaker id of test and dev have no overlap
    def __get_speaker_dict__(self, root_directory, dataset_type_name):
        if dataset_type_name == 'dev':
            raise Exception('voxceleb2 only contains train dataset and test dataset')

        if dataset_type_name == 'train':
            dev_path = os.path.join(root_directory, 'dev/aac/')
        else:
            dev_path = os.path.join(root_directory, 'test/aac/')

        speaker_dict = dict()
        for speaker_id in os.listdir(dev_path):
            path = os.path.join(dev_path, speaker_id)
            for p_dir, dirs, files in os.walk(path):
                m4a_files = set(map(lambda fname: os.path.join(p_dir, fname), files))
                m4a_files = set(filter(lambda fpath: fpath.endswith('.ogg') and self.is_valid_audio(fpath), m4a_files))
                if len(m4a_files) > 0:
                    speaker_dict[speaker_id] = speaker_dict.get(speaker_id, set()) | m4a_files

        return speaker_dict
