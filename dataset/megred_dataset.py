# -*- coding:utf-8 -*-

from dataset.basic_dataset import BasicDataset


class MergedDataset(BasicDataset):

    def __get_speaker_dict__(self, root_directory, dataset_type_name):
        if self.speaker_dict_list is None or len(self.speaker_dict_list) <= 0:
            raise Exception('speaker_dict_list is empty')

        speaker_dict_all = dict()
        for i, speaker_dict in enumerate(self.speaker_dict_list):
            speaker_dict_keys = list(sorted(speaker_dict.keys()))
            for speaker_id in speaker_dict_keys:
                path_set = speaker_dict[speaker_id]
                new_speaker_id = '#%d_%s' % (i, speaker_id)
                speaker_dict_all[new_speaker_id] = speaker_dict_all.get(new_speaker_id, set()) | path_set

        return speaker_dict_all
