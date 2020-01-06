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
    n_mels = 60
    # 大量实验表明,在语音特征中加入表征语音动态特征的查分参数(即:使用1阶2阶差分), 能够提高系统的识别性能
    used_delta_orders = (1, 2,)
    # 音频文件最短长度
    audio_file_min_duration = 3.0
    # 使用的长度
    used_duration = 2.0
    # 是否做数据增强
    do_augmentation = True
    # 是否固定锚点
    fixed_anchor = False
    # data loader 进程数
    num_workers = 10
    # batch size
    # 64: pre 6g.
    batch_size = 16
    # data loader 超时时间
    dataloader_timeout = 120
    # data loader是否进行shuffle
    shuffle = False
    #
    pin_memory = False
    # 是否使用gpu
    use_gpu = True

    # 训练最多的epoch数
    max_epoch = 200
    # 学习速率
    lr = 1e-3
    # SGD参数
    weight_decay = 5e-4
    # SGD参数
    momentum = 0.9
    # dropout
    dropout_keep_prop = 0.8
    # 是否冰冻参数
    em_train_fix_params = False

    # da domain_adapation参数
    da_lr = 5e-4
    da_weight_decay = 5e-4
    da_momentum = 0.9
    da_patience = 3
    # domain adapation loss 项权重 (normal_loss + lambda*da_loss)
    da_avg_acc_th = 0.55
    da_every_step = 2
    da_lambda = 1.0

    # 打印时机
    print_every_step = 100
    # 保存时机(0-1.0], 默认每个epoch保存两次(1/2处及epoch结尾处)
    save_points_list = [0.5, 1.0]
    # last_checkpoint.pth 保存间隔(单位step)
    last_checkpoint_save_interval = 200

    # 0:not search hard negative 1:hardest negatives  2:semi-hard negatives
    hard_negative_level = 2
    hard_negative_size = 3072  # 存储的历史数据总条数, 在这些数据中进行hard negative search
    hard_negative_recompute_every_step = 16

    # triplet loss
    triplet_loss_margin = 0.2
    # 类内半径
    clustering_every_step = 8
    intra_class_radius = 0.6

    # 覆盖网络参数
    override_net_params = False
    # 预训练文件路径, 如果不为空(预训练或正式训练会先加载该网络参数)
    pre_train_status_dict_path = '/home/mqb/project/Deep-Speaker-Using-Domain-Adaptive/net_data/checkpoints/pre_train/1_36442_2020-01-06_00_53_01.pth'
    # 正式训练文件路径, 如果不为空, 正式训练会先加载该网络参数
    status_dict_path = None

    # 图标大小(tensor board 中使用)
    icon_size = 15
    # 图标透明度
    icon_alpha = 0.35

    # train: 2338 speakers, 281241 files
    # test:  88 speakers, 19920 files
    libriSpeech = '/home/mqb/data/open_source_dataset/LibriSpeech'
    # 855 speakers, 102600 files, each speaker has 120 utterances
    # train: 0.9*855=769.5
    # dev:   0.05*855=42.75
    # test:  0.05*855=42.75
    st_cmds_20170001_1 = '/home/mqb/data/open_source_dataset/ST-CMDS-20170001_1-OS'
    # train: 5994 speakers, 1092009 files
    # test:  118 speakers, 36237 files
    voxceleb2 = '/home/mqb/data/open_source_dataset/vox2'
    # train: 1211 speakers, 148642 files.
    # test:  40 speakers, 4874 files
    voxceleb1 = '/home/mqb/data/open_source_dataset/vox1'
    # 测试数据集(vox1 子集)
    test_used_dataset = '/home/mqb/data/open_source_dataset/test'


opt = DefaultConfig()

# .m4a file, mono, 320k
# 60G	/data/open_source_dataset/LibriSpeech
# 13G	/data/open_source_dataset/ST-CMDS-20170001_1-OS
# 6.2G	/data/open_source_dataset/vox1
# 43G	/data/open_source_dataset/vox2
