# -*- coding:utf-8 -*-


import numpy as np
import time
import pickle
from utils.lmdb_util import LMDBUtil

class A(object):
    pass

if __name__ == '__main__':
    file_path = './test.lmdb'
    # lmdb_obj = LMDBUtil(file_path)
    lmdb_obj = LMDBUtil(file_path, readonly=True)

    # for i in range(100000):
    #     lmdb_obj.insert(i, np.random.randn(10, 128))
    #
    # for i in range(100000):
    #     lmdb_obj.update(i, np.random.randn(10, 128))
    #
    # for i in range(5):
    #     lmdb_obj.delete(i)

    t1 = time.time()
    a = 0
    for i in range(100000 - 5):
        np_obj = lmdb_obj.get(i + 5)
        # np_obj = lmdb_obj.get_use_fixed_txn(i + 5)
        a += np_obj.shape[0]
    t2 = time.time()
    print(t2 - t1)

    a = A()
    a.text = '123'
    a.lmdb = lmdb_obj
    with open('./test.lmdb/aaa.lmdb', 'wb') as f:
        del a.lmdb
        pickle.dump(a, f)



    # for i in range(100000 - 5):
    #     data = np.random.randn(10, 128)
    #     with open('./test.lmdb/%s.data' % i, 'wb') as f:
    #         pickle.dump(data, f)

    # t1 = time.time()
    # a = 0
    # for i in range(100000 - 5):
    #     with open('./test.lmdb/%s.data' % i, 'rb') as f:
    #         np_obj = pickle.load(f)
    #         a += np_obj.shape[0]
    # t2 = time.time()
    # print(t2 - t1)
