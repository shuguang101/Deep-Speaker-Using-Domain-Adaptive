# -*- coding:utf-8 -*-


import lmdb
import pickle


class LMDBUtil(object):

    def __init__(self, file_path, map_size=1024 ** 4, readonly=False, lock=True):
        self.file_path = file_path
        self.map_size = map_size
        self.readonly = readonly
        self.lock = lock

        self.env = lmdb.open(file_path, map_size=map_size, readonly=readonly, lock=lock)
        self.query_txn = None

    def put(self, key: str, value):
        assert value is not None
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
        result = self.put(key, value)
        return result

    def get(self, key: str):
        key = str(key).encode('utf-8')

        txn = self.env.begin(write=False)
        result = txn.get(key, default=None)
        if result is not None:
            result = pickle.loads(result)
        return result

    def get_use_fixed_txn(self, key: str):
        key = str(key).encode('utf-8')

        if self.query_txn is None:
            self.query_txn = self.env.begin(write=False)
        result = self.query_txn.get(key, default=None)
        if result is not None:
            result = pickle.loads(result)
        return result

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        self.query_txn = None
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['env']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        # Restore the previously opened file's state. To do so, we need to
        # reopen it and read from it until the line count is restored.
        self.env = lmdb.open(self.file_path,
                             map_size=self.map_size,
                             readonly=self.readonly,
                             lock=self.lock)
