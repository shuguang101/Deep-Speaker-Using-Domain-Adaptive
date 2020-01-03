# -*- coding:utf-8 -*-

from utils import audio_util

if __name__ == '__main__':
    # /data/open_source_dataset/test/
    # /data/open_source_dataset/vox1/
    # /data/open_source_dataset/ST-CMDS-20170001_1-OS/
    # /data/open_source_dataset/LibriSpeech /LibriSpeech/

    root_dir = '/data/open_source_dataset/LibriSpeech /LibriSpeech/train-other-500/'
    raw_file_ext_tuples = ('.wav', '.flac', '.m4a', '.ogg')
    new_file_ext = '.m4a'
    sr = 16000
    channel = 1
    remove_raw_file = True

    print('converting the audio format, please waiting ......')
    total, failed_list = audio_util.convert_audio_files(root_dir,
                                                        raw_file_ext_tuples,
                                                        new_file_ext,
                                                        sr, channel,
                                                        remove_raw_file)

    print('total', total)
    print('failed_list', failed_list)
