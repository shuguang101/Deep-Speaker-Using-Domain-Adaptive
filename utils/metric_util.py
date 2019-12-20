# -*- coding:utf-8 -*-

import random
import numpy as np
import torch
import torch.optim
from scipy.interpolate import interp1d
from scipy.optimize import brentq


def compute_equal_error_rate(speaker_net, thresholds_list, test_dataset, device,
                             max_people_to_compare=3, max_bs=8, min_bs=3):
    try:
        # 腾讯声纹识别 Ivector PLDA eer=0.7%
        # releases all unoccupied cached memory currently held by the caching allocator
        torch.cuda.empty_cache()

        # 阈值列表
        th_arr = np.array(thresholds_list, dtype=np.float32)

        # false rejection rate, 拒识率:相同指纹被误识别为不同指纹的比例. ngra:类内测试总次数
        frr_arr = np.zeros((len(thresholds_list),), dtype=np.float32)
        ngra_arr = np.zeros((len(thresholds_list),), dtype=np.float32)

        # false acceptance rate, ?误识率:不同指纹被误识别为是相同指纹的比例. nira:类间测试总次数
        far_arr = np.zeros((len(thresholds_list),), dtype=np.float32)
        nira_arr = np.zeros((len(thresholds_list),), dtype=np.float32)

        # eval mode
        speaker_net.eval()
        with torch.no_grad():
            speakers_num = test_dataset.num_of_speakers
            for nid_i in range(speakers_num):
                data_i = test_dataset.get_speaker_data_by_nid(nid_i, max_bs=max_bs, min_bs=min_bs)
                if data_i is None:
                    continue
                data_i_output = speaker_net(data_i.to(device)).detach().cpu().numpy()
                dis_i = np.linalg.norm(data_i_output[:-1] - data_i_output[1:], 2, axis=1)

                ngra_arr[:] = ngra_arr + dis_i.shape[0]
                compare_i = dis_i.reshape(-1, 1).repeat(len(thresholds_list), axis=1) > th_arr
                frr_arr[:] = frr_arr + np.sum(compare_i, axis=0)

                diff_nid_list = list(range(nid_i + 1, speakers_num))
                for nid_j in random.sample(diff_nid_list, min(len(diff_nid_list), max_people_to_compare)):
                    data_j = test_dataset.get_speaker_data_by_nid(nid_j, max_bs=max_bs, min_bs=min_bs)
                    if data_j is None:
                        continue
                    data_j_output = speaker_net(data_j.to(device)).detach().cpu().numpy()
                    dis_ij = np.linalg.norm(data_i_output[:min_bs] - data_j_output[:min_bs], 2, axis=1)

                    nira_arr[:] = nira_arr + dis_ij.shape[0]
                    compare_ij = dis_ij.reshape(-1, 1).repeat(len(thresholds_list), axis=1) <= th_arr
                    far_arr[:] = far_arr + np.sum(compare_ij, axis=0)

        frr_arr = frr_arr / ngra_arr
        far_arr = far_arr / nira_arr

        f1 = interp1d(th_arr, frr_arr)
        f2 = interp1d(th_arr, far_arr)
        f3 = lambda x: f1(x) - f2(x)

        th_root = brentq(f3, th_arr[0], th_arr[-1])
        eer = f1(th_root)

        return th_arr, frr_arr, far_arr, float(eer), float(th_root)
    except Exception:
        return -1, -1, -1, -1, -1
