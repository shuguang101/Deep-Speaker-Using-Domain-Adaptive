# -*- coding:utf-8 -*-


class DefaultConfig(object):
    ######################################################################
    # audio signal parameters
    ######################################################################

    # Nyquist定理(奈奎斯特): 如果想要从数字信号无损转到模拟信号，我们需要以最高信号频率的2倍的采样频率进行采样。
    # 通常人的声音的频率大概在3kHz~4kHz ，因此语音识别通常使用8k或者16k的wav提取特征。
    # 16kHz采样率的音频，傅里叶变换之后的频率范围为0-8KHz。
    # 人耳频率感受范围: 20Hz-20000Hz

    # Target Sampling rate, related to the Nyquist conditions, which affects
    # the range frequencies we can detect. By using STFT function,
    # the max value of freqs is sr//2.
    sr = 16000
    # Size of the FFT window, affects frequency granularity.
    # through experiments: n_freqs = n_fft//2 + 1
    # resolution: sr/n_fft, freqs=[0, sr/n_fft, 2*sr/n_fft, ...,  sr/2]
    # STFT return data shape: (n_freqs, nframes)
    n_fft = 2048
    # window size is typical 25ms
    win_length = 400
    # 参考overlap. Higher overlap will allow a higher granularity of offset
    # matching, but potentially more fingerprints.
    # When overlap_ratio = 0.5.
    # For librosa.stft(center=false) (i.e. mlab.specgram(pad_to=None)),
    # the returned nframes is floor((sr-n_fft)/step) + 1
    hop_length = 100  # typical 1/4
    # 窗口类型
    window = 'blackman'
    # mel滤波器数量
    n_mels = 128
    # 大量实验表明,在语音特征中加入表征语音动态特征的查分参数(即:使用1阶2阶差分), 能够提高系统的识别性能
    used_delta_orders = (1, 2)

    # 图标大小(tensor board 中使用)
    icon_size = 15
    # 图标透明度
    icon_alpha = 0.35

    # 音频文件最短长度
    audio_file_min_duration = 3.0
    # 使用的长度
    used_duration = 2.0
    # 是否做数据增强
    do_augmentation = True
    # 是否固定锚点
    fixed_anchor = False

    use_gpu = True

    max_epoch = 200
    lr = 1e-3
    weight_decay = 5e-4
    momentum = 0.9
    dropout_keep_prop = 0.75

    pre_train_status_dict_path = None
    # 覆盖网络参数
    override_net_params = False

    print_every_step = 100

    # 保存时机(0-1.0], 默认每个epoch保存两次(1/2处及epoch结尾处)
    save_points_list = [0.5, 1.0]
    # last_checkpoint.pth 保存间隔(单位step)
    last_checkpoint_save_interval = 200

    num_workers = 2  # 10
    pin_memory = False
    batch_size = 8  # 13
    dataloader_timeout = 120
    shuffle = False

    # train: 2338 speakers, 281241 files
    # test:  88 speakers, 19920 files
    libriSpeech = '/data/open_source_dataset/LibriSpeech'
    # 855 speakers, 102600 files, each speaker has 120 utterances
    # train: 0.9*855=769.5
    # dev:   0.05*855=42.75
    # test:  0.05*855=42.75
    st_cmds_20170001_1 = '/data/open_source_dataset/ST-CMDS-20170001_1-OS'
    # train: 5994 speakers, 1092009 files
    # test:  118 speakers, 36237 files
    voxceleb2 = '/data/open_source_dataset/vox2'
    # train: 1211 speakers, 148642 files.
    # test:  40 speakers, 4874 files
    voxceleb1 = '/data/open_source_dataset/vox1'
    # 测试数据集(vox1 子集)
    test_used_dataset = '/data/open_source_dataset/test'

    #
    # vis = True
    # embedding_every_step = -1

    # # schedule_lr_decay_every_step = 70000
    # # lr_decay_every_step = 5000  # 500
    # # decay_error_tolerance = 1.1  # do decay if avg_loss > previous_loss * decay_error_tolerance
    # save_every_step = [1, 2]
    # clustering_every_step = -1  # 8
    #
    # # eval_every_step = 37868
    #
    # em_train_fix_params = False
    #
    # triplet_loss_margin = 0.2
    # intra_class_radius = 0.6
    #
    # max_epoch = 200
    #
    # # off line search, on line search need bigger batch size
    # search_hard_negative = True
    # hard_negative_cache_size = 250  # 500
    # hard_negative_level = 0  # 0:hardest negatives  1:semi-hard negatives  2:no-zero loss negatives
    # hard_negative_recompute_every_step = 5
    # fixed_anchor = False
    # do_augmentation = True
    #
    # status_dict_path = '/home/mqb/project/DeepSpeaker/checkpoints/cnn/26_2924856__2019-10-18_19_56_13.pth'
    # pre_train_model_path = None
    #

    # dataloader_timeout = 120
    # restart_retry_num = 3  # when python program failed, restart it
    # restart_retry_num_reset_time = 60 * 30  # when program runs normally for 30 minutes, reset retry_num_flag
    #

    #
    # summary_log_dir = './summary_log_dir/cnn'
    # checkpoints_dir = './checkpoints/cnn'
    #
    # summary_log_dir_pre = './summary_log_dir/pre_train/cnn'
    # checkpoints_dir_pre = './checkpoints/pre_train/cnn'
    #
    # shuffle = False
    #
    # # overwritten_val_in_status_dict
    # ow_val = False
    #
    # def is_do_save(self, step, total_batch_step):
    #     sign = False
    #     for e_step in self.save_every_step:
    #         tmp = int(total_batch_step // e_step)
    #         if step % tmp == 0:
    #             sign = True
    #             break
    #     return sign


opt = DefaultConfig()

# 60G	/data/open_source_dataset/LibriSpeech
# 13G	/data/open_source_dataset/ST-CMDS-20170001_1-OS
# 6.2G	/data/open_source_dataset/vox1
# 43G	/data/open_source_dataset/vox2
