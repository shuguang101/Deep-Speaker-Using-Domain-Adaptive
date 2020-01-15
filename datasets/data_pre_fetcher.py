# -*- coding:utf-8 -*-

import torch


class DataPreFetcher(object):
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.batch = []
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
            if self.batch is None:
                return
        except StopIteration:
            self.batch = None
            return

        with torch.cuda.stream(self.stream):
            for index, val in enumerate(self.batch):
                self.batch[index] = self.batch[index].to(device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            self.preload()
        return batch
