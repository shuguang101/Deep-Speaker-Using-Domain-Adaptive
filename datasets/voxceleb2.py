# -*- coding:utf-8 -*-

import os
from datasets.basic_dataset import BasicDataset


class VoxCeleb2(BasicDataset):
    ext_tuples = ('.wav', '.ogg', '.flac', '.m4a')
    sid_pre = 'vox2'  # 保证id全局唯一, 添加前缀

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
                audio_files = set(map(lambda fname: os.path.join(p_dir, fname), files))
                audio_files = set(filter(lambda fpath: self.is_in_exts(fpath, self.ext_tuples)
                                                       and self.is_valid_audio(fpath), audio_files))
                if len(audio_files) > 0:
                    speaker_id_with_pre = self.sid_pre + speaker_id
                    speaker_dict[speaker_id_with_pre] = speaker_dict.get(speaker_id_with_pre, set()) | audio_files

        return speaker_dict
