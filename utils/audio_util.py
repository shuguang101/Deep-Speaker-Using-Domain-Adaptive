# -*- coding:utf-8 -*-

import librosa
import random
import numpy as np
from essentia import standard
from scipy.signal import lfilter
import scipy.signal as signal


def load_audio_as_mono(audio_path, output_sample_rate):
    """
        When load a audio file, essentia is faster than soundfile,
        and soundfile is much faster then librosa.
    """

    mono_loader = standard.MonoLoader(filename=audio_path, sampleRate=output_sample_rate)
    y = mono_loader()

    return y


def de_noise(audio_data, audio_sr):
    # 滤波去除噪音

    # 维纳滤波器去噪,时域
    new_audio_data = signal.wiener(audio_data)

    # 趋势消除
    new_audio_data = signal.detrend(new_audio_data)

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

    spec_data = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length,
                             window=window, win_length=win_length, center=True)

    # 获取幅值频谱及相位
    magnitude, phase = np.abs(spec_data), np.angle(spec_data)

    # 应用mel滤波器
    mel_basis = librosa.filters.mel(audio_sr, n_fft, n_mels=256)
    mel_spec = np.dot(mel_basis, magnitude ** 2)

    # 获取fbank特征, shape=[n_mels, t]
    f_bank = librosa.power_to_db(mel_spec)

    # 标准化
    f_bank = (f_bank - np.mean(f_bank, axis=0)) / (np.std(f_bank, axis=0) + 2e-12)

    return f_bank.T


def do_audio_augmentation(audio_data, audio_sr):
    # 添加噪音, power level of the noise generator [dB], (负无穷, 0]
    noise_adder = standard.NoiseAdder(level=-100)
    new_audio_data = noise_adder(audio_data)

    # If rate > 1, then the signal is speed up.
    # If rate < 1, then the signal is slowed down.
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
