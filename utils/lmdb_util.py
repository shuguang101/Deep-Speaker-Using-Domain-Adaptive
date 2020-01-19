# -*- coding:utf-8 -*-


import lmdb
import pickle


class LMDBUtil(object):

    def __init__(self, file_path, map_size=1024 ** 4, readonly=False, lock=True):
        self.env = lmdb.open(file_path, map_size=map_size, readonly=readonly, lock=lock)
        self.query_txn = None

    def insert(self, key: str, value):
        key = str(key).encode('utf-8')
        value = pickle.dumps(value)

        txn = self.env.begin(write=True)
        result = txn.put(key, value)
        txn.commit()
        return result

    def delete(self, key: str):
        key = str(key).encode('utf-8')

        txn = self.env.begin(write=True)
        result = txn.delete(key)
        txn.commit()
        return result

    def update(self, key: str, value):
        result = self.insert(key, value)
        return result

    def get(self, key: str, default=None):
        key = str(key).encode('utf-8')

        txn = self.env.begin(write=False)
        result = txn.get(key, default=default)
        result = pickle.loads(result)
        return result

    def get_use_fixed_txn(self, key: str, default=None):
        key = str(key).encode('utf-8')

        if self.query_txn is None:
            self.query_txn = self.env.begin(write=False)
        result = self.query_txn.get(key, default=default)
        result = pickle.loads(result)
        return result
