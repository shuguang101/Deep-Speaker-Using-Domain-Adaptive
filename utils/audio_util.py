# -*- coding:utf-8 -*-

import essentia
import librosa
import random
import os
import subprocess
import numpy as np
import scipy.signal as signal
import traceback
from scipy.signal import lfilter

from numpy import (asarray, mean, ndarray, ones, product, ravel, where)
from scipy.signal.signaltools import correlate

# 屏蔽日志输出
essentia.log.infoActive = False
from essentia import standard


def load_audio_as_mono(audio_path, output_sample_rate):
    """
        When load a audio file, essentia is faster than soundfile,
        and soundfile is much faster then librosa.
    """

    mono_loader = standard.MonoLoader(filename=audio_path, sampleRate=output_sample_rate)
    y = mono_loader().astype(np.float32)

    return y


def de_noise(audio_data, audio_sr):
    # 滤波去除噪音

    # 维纳滤波器去噪,时域
    new_audio_data = wiener_my(audio_data)

    # 趋势消除
    new_audio_data = signal.detrend(new_audio_data).astype(np.float32)

    # 去除直流成分, 高通滤波器
    dc_removal = standard.DCRemoval(cutoffFrequency=40, sampleRate=audio_sr)
    new_audio_data = dc_removal(new_audio_data)

    # 语音信号预加重，预增强系数范围为[0,1),常用值0.97
    # 对高频部分进行加重，去除口唇辐射影响, 增加高频分辨率
    new_audio_data = pre_emphasis_filter(new_audio_data)

    return new_audio_data


def fbank(audio_data, audio_sr, **kwargs):
    n_fft = kwargs.get('n_fft', 2048)
    win_length = kwargs.get('win_length', 1103)  # 25ms (sr=44100)
    hop_length = kwargs.get('hop_length', 275)  # 1/4 of window size
    window = kwargs.get('window', 'blackman')
    n_mels = kwargs.get('n_mels', 256)

    spec_data = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length,
                             window=window, win_length=win_length, center=False)

    # 获取幅值频谱及相位
    magnitude, phase = np.abs(spec_data), np.angle(spec_data)

    # 应用mel滤波器
    mel_basis = librosa.filters.mel(audio_sr, n_fft, n_mels=n_mels)
    mel_spec = np.dot(mel_basis, magnitude ** 2)

    # 获取fbank特征, shape=[n_mels, t]
    f_bank = librosa.power_to_db(mel_spec)
    f_bank = f_bank.astype(np.float32)

    return f_bank.T


def add_deltas_librosa(data, orders=(1, 2), axis=0, width=5):
    # 大量实验表明,在语音特征中加入表征语音动态特征的查分参数, 能够提高系统的识别性能
    # data shape: [time_dim, features_dim]
    diff_list = [data]
    for i in orders:
        # default axis=-1, return delta_data: shape=(d, t)]
        delta_i = librosa.feature.delta(data, width=width, order=i, axis=axis)
        diff_list.append(delta_i)
    result_data = np.concatenate(tuple(diff_list), axis=1)
    return result_data


def do_audio_augmentation(audio_data, audio_sr):
    # 添加噪音, power level of the noise generator [dB], (负无穷, 0]
    noise_adder = standard.NoiseAdder(level=-100)
    new_audio_data = noise_adder(audio_data)

    # If rate > 1, then the signal is speed up.
    # If rate < 1, then the signal is slowed down.
    # return: y_stretch : np.ndarray [shape=(rate * n,)]
    new_audio_data = librosa.effects.time_stretch(new_audio_data, random.randint(95, 105) * 0.01)

    # pitch_shift
    # 两音高度上的距离,叫做音程(Interval)
    # 发生在自然音阶中的半音,两音的名称不同, 叫做自然半音(Diatonic Semitone).
    # 发生在半音阶(即变化音阶)中的半音,两 音的名称相同,叫做变化半音 (Chromatic Semitone).
    # 不同的是自然半音：音名相邻C D E F G A B如果这些音名相邻比如 E-F 就是自然的 #C-D 也是.
    # 变化半音 C-#C （同一音级） #D-bF（相隔一个音级）(没有按照C D E F G A B的顺序相邻）

    # 简谱里有12345671,前一个1和后一个1在听觉上是同一个音,只是在调门上相差8度而已
    # 1      2         3     4       5      6     7   i
    #   #1      #2               #4     #5    #6

    # A pitch shifter is a sound effects unit that raises or lowers the pitch of an audio signal by a preset interval.
    # For example, a pitch shifter set to increase the pitch by a fourth
    # will raise each note three diatonic intervals above the notes actually played.
    # Simple pitch shifters raise or lower the pitch by one or two octaves,
    # while more sophisticated devices offer a range of interval alterations.
    # Pitch shifters are included in most audio processors today.

    # 音调的学名：CDEFGAB 应的是：哆来咪发索拉索. 音高越高声音越尖,音高越低声音越低沉.
    # 一般男性的音域从大字组C1-C5,男性比女性在生理上音域低一个八度. 女人由于声带短小,天性可以唱的更高,音域为C2-B5
    # 音域从低到高划分如下，高音自然也按照这个标准来划分:
    # A1 B1 C D E F G A B c d e f g a b c1 d1 e1 f1 g1 a1 b1 c2 d2 e2 f2 g2 a2 b2 c3 d3 e3 f3 g3
    # a3 b3 c4 d4 e4 f4 g4 a4 b4 c5 d5 e5 f5 g5 a5 b5 c6......
    n_steps = 0.6 * random.randint(-10, 10)
    new_audio_data = librosa.effects.pitch_shift(new_audio_data, audio_sr, n_steps, 12)

    return new_audio_data


def pre_emphasis_filter(sin, alpha=0.97):
    # 预加重
    assert sin.ndim == 1
    s_out = lfilter([1 - alpha], 1, sin)  # preemphasis filtering
    return s_out


def rm_dc_n_dither(sin, sr):
    """
    refer to: https://github.com/a-nagrani/VGGVox/blob/master/mfcc/rm_dc_n_dither.m
    """

    assert (sr == 16000 or sr == 8000) and sin.ndim == 1

    alpha = 0.99 if sr == 16000 else 0.999

    sin = lfilter([1, -1], [1, -alpha], sin)  # remove dc
    dither = np.random.rand(*sin.shape) + np.random.rand(*sin.shape) - 1
    spow = np.std(sin)

    s_out = sin + 1e-6 * spow * dither

    return s_out


def wiener_my(im, mysize=None, noise=None):
    # copied from signaltools.py, and fixed the 'divide by zero encountered' problem.

    """
    Perform a Wiener filter on an N-dimensional array.

    Apply a Wiener filter to the N-dimensional array `im`.

    Parameters
    ----------
    im : ndarray
        An N-dimensional array.
    mysize : int or array_like, optional
        A scalar or an N-length list giving the size of the Wiener filter
        window in each dimension.  Elements of mysize should be odd.
        If mysize is a scalar, then this scalar is used as the size
        in each dimension.
    noise : float, optional
        The noise-power to use. If None, then noise is estimated as the
        average of the local variance of the input.

    Returns
    -------
    out : ndarray
        Wiener filtered result with the same shape as `im`.

    """
    im = asarray(im)
    if mysize is None:
        mysize = [3] * im.ndim
    mysize = asarray(mysize)
    if mysize.shape == ():
        mysize = np.repeat(mysize.item(), im.ndim)

    # Estimate the local mean
    lMean = correlate(im, ones(mysize), 'same') / product(mysize, axis=0)

    # Estimate the local variance
    lVar = (correlate(im ** 2, ones(mysize), 'same') /
            product(mysize, axis=0) - lMean ** 2)

    # 防止 除零 错误
    lVar += np.finfo(np.float32).eps

    # Estimate the noise power if needed.
    if noise is None:
        noise = mean(ravel(lVar), axis=0)

    res = (im - lMean)
    res *= (1 - noise / lVar)
    res += lMean
    out = where(lVar < noise, lMean, res)

    return out


def convert_audio_files(root_dir,
                        raw_file_ext_tuples=('.wav',),
                        new_file_ext='.m4a',
                        sr=44100,
                        channel=1,
                        remove_raw_file=False):
    """
    使用ffmpeg转换音频文件格式及采样率
    """

    # 使用ffmpeg转换audio文件
    def audio_convert(raw_audio_path):
        result = 0
        try:
            audio_path_without_ext = os.path.splitext(raw_audio_path)[0]
            new_audio_path = str(audio_path_without_ext + new_file_ext)
            if new_audio_path == raw_audio_path:
                new_audio_path += new_file_ext

            cmd_f = str(new_file_ext[1:])
            if cmd_f == 'm4a':
                cmd_f = 'mp4'

            cmd = [
                'ffmpeg',
                '-i', str(raw_audio_path),
                '-f', cmd_f,
                '-ar', str(sr),
                '-ac', str(channel),
                '-y', new_audio_path

            ]

            if cmd_f not in ('flac', 'wav', 'ape', 'ogg'):
                cmd.insert(3, '-ab')
                cmd.insert(4, '320k')

            p1 = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p1.communicate()
            stdout_str, stderr_str = stdout.decode('utf-8'), stderr.decode('utf-8')

            if p1.returncode != 0:
                print('audio convert error: ', raw_audio_path)
                print('std out:\r\n%s' % stdout_str)
                print('std error:\r\n%s' % stderr_str)
                result = -1
        except Exception:
            result = -1
        return result

    total = 0
    failed_list = []

    try:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file_path)[-1]
                if file_ext in raw_file_ext_tuples:
                    is_success = audio_convert(file_path)
                    if is_success == 0:
                        if remove_raw_file:
                            os.remove(file_path)
                    else:
                        failed_list.append(file_path)
                    total += 1
    except Exception:
        print(traceback.format_exc())

    return total, failed_list
