# -*- coding:utf-8 -*-

from datasets.basic_dataset import BasicDataset


class MergedDataset(BasicDataset):

    def __get_speaker_dict__(self, root_directory, dataset_type_name):
        if self.speaker_dict_list is None or len(self.speaker_dict_list) <= 0:
            raise Exception('speaker_dict_list is empty')

        speaker_dict_all = dict()
        for i, speaker_dict in enumerate(self.speaker_dict_list):
            speaker_dict_keys = list(sorted(speaker_dict.keys()))
            for speaker_id in speaker_dict_keys:
                path_set = speaker_dict[speaker_id]
                speaker_dict_all[speaker_id] = speaker_dict_all.get(speaker_id, set()) | path_set
                # 原始说话人id -> 域id
                self.sid2did_dict[speaker_id] = i
        return speaker_dict_all
