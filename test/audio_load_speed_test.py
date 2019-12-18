# -*- coding:utf-8 -*-

import time
import numpy as np
from utils import audio_util

if __name__ == '__main__':
    audio_path = '../audios/20170001P00091A0063.ogg'  # sr=16000
    sr = 48000

    t1 = time.time()
    y = audio_util.load_audio_as_mono(audio_path, sr)
    t2 = time.time()
    print('y.shape: %s, %s load time cost: %s, duration: %s' % (y.shape, y[:3], t2 - t1, y.shape[0] / sr))

    count = 100
    t1 = time.time()
    for i in range(count):
        y = audio_util.load_audio_as_mono(audio_path, sr)
    t2 = time.time()
    print('times: %s, avg load time: %s' % (count, (t2 - t1) / count))

    fbank = audio_util.fbank(y, sr)
    print('fbank', fbank.shape, fbank[:3, 1])

    y_noise = audio_util.de_noise(y, sr)
    print('y_noise', y_noise.shape, y_noise[:3])

    y_augmentation = audio_util.do_audio_augmentation(y, sr)
    print('y_augmentation', y_augmentation.shape, y_augmentation[:3])
