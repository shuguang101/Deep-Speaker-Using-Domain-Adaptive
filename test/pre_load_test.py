# -*- coding:utf-8 -*-


import torch
import time

from torch.utils.data import DataLoader
from config import opt
from utils import common_util
from datasets.voxceleb1 import VoxCeleb1
from datasets.data_pre_fetcher import DataPreFetcher
from datasets.my_dataloader import DataLoaderX, NoBlockDataLoader

if __name__ == '__main__':
    opt_attrs = common_util.get_all_attribute(opt)
    params_dict = {k: getattr(opt, k) for k in opt_attrs}
    # params_dict['sr'] = 44100

    root_directory = '/home/mqb/data/open_source_dataset/test'
    params_dict['dataset_type_name'] = 'test'
    vox1 = VoxCeleb1(root_directory, **params_dict)

    device = torch.device('cuda')

    t1 = time.time()
    dl = DataLoader(vox1, batch_size=16, num_workers=3)
    # dl = DataLoaderX(vox1, batch_size=16, num_workers=3)
    # dl = NoBlockDataLoader(vox1, batch_size=16, num_workers=3)

    for ii, (y_a, y_p, y_n, p_nid, p_did, n_nid, n_did) in enumerate(dl):
        print(ii, y_a.sum().item(), y_a.shape, y_a, flush=True)
    t2 = time.time()
    print(t2 - t1)
