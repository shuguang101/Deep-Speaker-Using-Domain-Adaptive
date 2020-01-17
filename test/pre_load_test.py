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

    root_directory = '/home/mqb/data/open_source_dataset/vox1'
    params_dict['dataset_type_name'] = 'test'
    params_dict['feature_cache_root_dir'] = '/media/HDisk_2T/data/cache_data/'
    params_dict['do_feature_cache'] = True
    vox1 = VoxCeleb1(root_directory, **params_dict)

    device = torch.device('cuda')

    t1 = time.time()
    # dl = DataLoader(vox1, batch_size=256, num_workers=3)  # 68.2 8.22, 66.4
    # dl = DataLoaderX(vox1, batch_size=256, num_workers=3)  # 65.4 8.13, 63.4
    dl = NoBlockDataLoader(vox1, batch_size=256, num_workers=3)  # 68.3 7.9, 68.1

    print(vox1.do_feature_cache, vox1.feature_cache_root_dir, vox1.used_nframe)
    total = 0
    # for i in range(2):
    #     for ii, (y_a, y_p, y_n, p_nid, p_did, n_nid, n_did) in enumerate(dl):
    #         y_a = y_a.to(device)
    #         y_p = y_p.to(device)
    #         y_n = y_n.to(device)
    #         p_nid = p_nid.to(device)
    #         p_did = p_did.to(device)
    #         n_nid = n_nid.to(device)
    #         n_did = n_did.to(device)
    #         total += y_a.shape[0]
    # t2 = time.time()
    # print(t2 - t1, flush=True)
    #
    # t1 = time.time()
    for i in range(2):
        pre_fetcher = DataPreFetcher(dl, device)
        batch = pre_fetcher.next()
        ii = 0
        while batch is not None:
            y_a, y_p, y_n, p_nid, p_did, n_nid, n_did = batch
            total += y_a.shape[0]
            batch = pre_fetcher.next()
            ii += 1
    t2 = time.time()
    print(t2 - t1, flush=True)
