# -*- coding:utf-8 -*-

import multiprocessing
import time

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from prefetch_generator import BackgroundGenerator
from datasets.sliced_dataset import SlicedDataset


class DataLoaderX(DataLoader):

    # https://github.com/justheuristic/prefetch_generator
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class NoBlockDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False,
                 sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=default_collate, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None,
                 each_worker_max_prefetch=5):

        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.each_worker_max_prefetch = each_worker_max_prefetch

        # raw params
        params_dict = {
            'batch_size': batch_size, 'shuffle': shuffle, 'sampler': sampler, 'batch_sampler': batch_sampler,
            'num_workers': num_workers, 'collate_fn': collate_fn, 'pin_memory': pin_memory,
            'drop_last': drop_last, 'timeout': timeout, 'worker_init_fn': worker_init_fn
        }
        # worker params
        worker_params_dict = {**params_dict, **{'num_workers': 1}}

        self.job_list = list()
        self.loaders_list = list()
        self.total_len = 0
        ds_len = self.dataset.__len__()
        num_workers = max(1, self.num_workers)
        num_workers = min(num_workers, max(1, int(ds_len / self.batch_size)))

        worker_size = int(ds_len / num_workers)
        start = 0
        for i in range(num_workers - 1):
            ds = SlicedDataset(self.dataset, start, start + worker_size)
            dl = DataLoader(ds, **worker_params_dict)
            self.total_len += dl.__len__()
            self.loaders_list.append(dl)
            start += worker_size

        ds = SlicedDataset(self.dataset, start, ds_len)
        dl = DataLoader(ds, **worker_params_dict)
        self.total_len += dl.__len__()
        self.loaders_list.append(dl)

        mgr = multiprocessing.Manager()
        self.queue = mgr.Queue()

        for dl in self.loaders_list:
            job = multiprocessing.Process(target=self.loader_worker,
                                          args=(dl, self.queue, self.each_worker_max_prefetch * len(self.loaders_list)))
            # job.daemon = True
            job.start()
            self.job_list.append(job)

    @staticmethod
    def loader_worker(dl, queue, cache_size):
        for batch in dl:
            while queue.qsize() > cache_size:
                time.sleep(0.1)
            queue.put(batch)

    def __iter__(self):
        for i in range(self.total_len):
            batch = self.queue.get()
            yield batch
        for job in self.job_list:
            job.terminate()

    def __len__(self):
        length = self.total_len
        return length
