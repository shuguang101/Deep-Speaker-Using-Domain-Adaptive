# -*- coding:utf-8 -*-


from torch.utils.data import Dataset


class SlicedDataset(Dataset):

    def __init__(self, ds_obj, start, end) -> None:
        assert isinstance(ds_obj, Dataset)
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert 0 <= start <= ds_obj.__len__() - 1
        assert start < end <= ds_obj.__len__()

        self.ds_obj = ds_obj
        # include start, not include end. [start, end)
        self.start = start
        self.end = end

    def __getitem__(self, index):
        new_index = self.start + index
        item = self.ds_obj.__getitem__(new_index)
        return item

    def __len__(self):
        new_len = self.end - self.start
        return new_len
