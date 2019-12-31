# -*- coding:utf-8 -*-

from datasets.basic_dataset import BasicDataset


class MergedDataset(BasicDataset):

    def __get_speaker_dict__(self, root_directory, dataset_type_name):
        if self.dataset_tuple_list is None or len(self.dataset_tuple_list) <= 0:
            raise Exception('dataset_tuple_list is empty')

        speaker_dict_all = dict()
        for i, dataset_i in enumerate(self.dataset_tuple_list):
            speaker_dict = dataset_i.speaker_dict
            for speaker_id in list(sorted(speaker_dict.keys())):
                path_set = set(speaker_dict[speaker_id])
                speaker_dict_all[speaker_id] = speaker_dict_all.get(speaker_id, set()) | path_set
                # 原始说话人id -> 域id
                self.sid2did_dict[speaker_id] = i

            speaker_dict = dataset_i.eval_used_dict
            for speaker_id in list(sorted(speaker_dict.keys())):
                path_set = set(speaker_dict[speaker_id])
                speaker_dict_all[speaker_id] = speaker_dict_all.get(speaker_id, set()) | path_set
                # 原始说话人id -> 域id
                self.sid2did_dict[speaker_id] = i
        return speaker_dict_all
