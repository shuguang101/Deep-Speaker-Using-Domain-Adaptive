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
        return BackgroundGenerator(super().__iter__(), 3)


class NoBlockGenerator(object):

    def __init__(self, no_block_dl) -> None:
        assert isinstance(no_block_dl, NoBlockDataLoader)

        self.dataset = no_block_dl.dataset
        self.num_workers = no_block_dl.num_workers
        self.batch_size = no_block_dl.batch_size
        self.each_worker_max_prefetch = no_block_dl.each_worker_max_prefetch

        # raw params
        params_dict = {
            'batch_size': no_block_dl.batch_size, 'shuffle': no_block_dl.shuffle,
            'sampler': no_block_dl.sampler, 'batch_sampler': no_block_dl.batch_sampler,
            'num_workers': no_block_dl.num_workers, 'collate_fn': no_block_dl.collate_fn,
            'pin_memory': no_block_dl.pin_memory, 'drop_last': no_block_dl.drop_last,
            'timeout': no_block_dl.timeout, 'worker_init_fn': no_block_dl.worker_init_fn
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
        self.current_index = 0

        mgr = multiprocessing.Manager()
        self.queue = mgr.Queue(maxsize=self.each_worker_max_prefetch * len(self.loaders_list))

        for dl in self.loaders_list:
            job = multiprocessing.Process(target=self.loader_worker,
                                          args=(dl, self.queue))
            # job.daemon = True
            job.start()
            self.job_list.append(job)

    @staticmethod
    def loader_worker(dl, queue):
        for batch in dl:
            queue.put(batch)

    def __next__(self):
        if self.current_index >= self.total_len:
            for job in self.job_list:
                job.terminate()
            raise StopIteration

        batch = self.queue.get()
        self.current_index += 1
        return batch

    def __iter__(self):
        return self


class NoBlockDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False,
                 sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=default_collate, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None,
                 each_worker_max_prefetch=3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.each_worker_max_prefetch = each_worker_max_prefetch

        self.no_block_generator = NoBlockGenerator(self)
        self.is_first = True

    def __iter__(self):
        if self.is_first:
            self.is_first = False
            return self.no_block_generator
        else:
            return NoBlockGenerator(self)

    def __len__(self):
        length = self.no_block_generator.total_len
        return length
