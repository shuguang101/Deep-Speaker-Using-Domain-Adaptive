# -*- coding:utf-8 -*-

import time
from utils import audio_util

if __name__ == '__main__':
    audio_path = '../audios/20170001P00091A0063.ogg'  # sr=16000
    sr = 48000

    t1 = time.time()
    y = audio_util.load_audio_as_mono(audio_path, sr)
    t2 = time.time()
    print('y.shape: %s, load time cost: %s, duration: %s' % (y.shape, t2 - t1, y.shape[0] / sr))

    fbank = audio_util.fbank(y, sr)
    print(fbank.shape, fbank[:,1])

    count = 100
    t1 = time.time()
    for i in range(count):
        y = audio_util.load_audio_as_mono(audio_path, sr)
    t2 = time.time()
    print('times: %s, avg load time: %s' % (count, (t2 - t1) / count))
