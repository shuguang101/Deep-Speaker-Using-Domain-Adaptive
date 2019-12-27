# -*- coding:utf-8 -*-


if __name__ == '__main__':
    # use 'spawn' in main file's first line, to prevent deadlock occur
    import multiprocessing

    multiprocessing.set_start_method('spawn')

import fire
import torch
import os
import datetime
import copy
import numpy as np
import random
from functools import reduce
from torchnet import meter
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config import opt
from utils import common_util
from nets.speaker_net_cnn import SpeakerNetEM
from nets.intra_class_loss import IntraClassLoss
from datasets.voxceleb1 import VoxCeleb1
from utils import metric_util


def do_net_eval(**kwargs):
    speaker_net = kwargs['speaker_net']
    device = kwargs['device']
    global_step = kwargs['global_step']
    summary_writer = kwargs['summary_writer']
    test_dataset = kwargs['test_dataset']

    # 清空缓存
    torch.cuda.empty_cache()
    # 切换到eval模式
    speaker_net.eval()

    thresholds_list = list(np.arange(0, 10, 0.001, dtype=np.float32))
    th_arr, frr_arr, far_arr, eer, th_root = metric_util.compute_equal_error_rate(speaker_net,
                                                                                  thresholds_list,
                                                                                  test_dataset,
                                                                                  device,
                                                                                  max_people_to_compare=5,
                                                                                  max_bs=32,
                                                                                  min_bs=16)

    summary_writer.add_scalar('eval_eer/eer', eer, global_step)
    summary_writer.add_scalar('eval_eer/th', th_root, global_step)

    # 测试完成切换回train模式
    speaker_net.train()


def train(**kwargs):
    # 获取该文件的目录
    fdir = os.path.split(os.path.abspath(__file__))[0]

    # 覆盖默认参数
    for k, v in kwargs.items():
        setattr(opt, k, v)

    # 获取参数
    opt_attrs = common_util.get_all_attribute(opt)
    params_dict = {k: getattr(opt, k) for k in opt_attrs}

    # 数据集参数
    dataset_train_param = {**params_dict, **{'dataset_type_name': 'train'}}
    dataset_test_param = {**params_dict, **{'dataset_type_name': 'test'}}

    # 读取训练数据
    train_dataset, train_dataloader = common_util.load_data(opt, **dataset_train_param)
    identity_speaker_data_gen = train_dataset.identity_speaker_data_gen_forever()
    test_dataset, test_dataloader = common_util.load_data(opt, **dataset_test_param)

    # tensor board summary writer
    summary_writer = SummaryWriter(log_dir=os.path.join(fdir, 'net_data/summary_log_dir/train/'))

    lr = opt.lr
    epoch, global_step = 1, 1
    device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')
    map_location = lambda storage, loc: storage.cuda(0) if opt.use_gpu else lambda storage, loc: storage

    speaker_net = SpeakerNetEM(device, opt.dropout_keep_prop)
    speaker_net.to(device)

    # 固定一部分参数
    if opt.em_train_fix_params:
        for param in speaker_net.conv_layer.parameters():
            param.requires_grad = False
        for param in speaker_net.conv_fc1.parameters():
            param.requires_grad = False
    else:
        for param in speaker_net.parameters():
            param.requires_grad = True

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, speaker_net.parameters()),
                                lr=lr,
                                weight_decay=opt.weight_decay,
                                momentum=opt.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           patience=1,
                                                           factor=0.2)

    triplet_loss = torch.nn.TripletMarginLoss(margin=opt.triplet_loss_margin, p=2).to(device)
    intra_class_loss = IntraClassLoss(opt.intra_class_radius).to(device)
    avg_loss_meter = meter.AverageValueMeter()

    if opt.pre_train_status_dict_path:
        print('load pre train model parameters')
        pre_trained_states_dict = torch.load(opt.pre_train_model_path, map_location)['net']

        model_state_dict = speaker_net.state_dict()
        part_of_pre_trained_states_dict = {k: v for k, v in pre_trained_states_dict.items() if
                                           k in model_state_dict}

        model_state_dict.update(part_of_pre_trained_states_dict)
        speaker_net.load_state_dict(model_state_dict)

        del pre_trained_states_dict
        torch.cuda.empty_cache()

    if opt.status_dict_path:
        status_dict = torch.load(opt.status_dict_path, map_location)
        speaker_net.load_state_dict(status_dict['net'])

        try:
            print('load status dict paramters')
            optimizer.load_state_dict(status_dict['optimizer'])
        except Exception as e:
            print('warning: load optimizer status dict failed\r\n', e)

        global_step = status_dict['global_step']
        epoch = status_dict['epoch']
        lr = status_dict['optimizer']['param_groups'][0]['lr']

        del status_dict
        torch.cuda.empty_cache()

    # 覆盖网络参数
    if opt.override_net_params:
        lr = opt.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        speaker_net.set_dropout_keep_prop(opt.dropout_keep_prop)

    # 存储历史数据, 用于hard negative search
    his_data_list = []
    his_label_list = []
    his_data, his_data_label, his_data_out = None, None, None

    speaker_net.train()
    while epoch <= opt.max_epoch:
        total_batch = train_dataloader.__len__()
        for i, (a, p, n, p_nid, p_did, n_nid, n_did) in enumerate(train_dataloader):
            # 清空cuda缓存
            torch.cuda.empty_cache()

            # run one step
            anchor = a.to(device)
            positive = p.to(device)
            negative = n.to(device)

            # hard negative search
            if opt.hard_negative_level > 0:
                his_data_list.append(p)
                his_label_list.append(p_nid)
                current_data_length = reduce(lambda x, y: x + y, map(lambda x: x.shape[0], his_data_list))
                if current_data_length > opt.hard_negative_size and len(his_data_list) > 1:
                    # index = random.choice(range(len(his_data_list)))
                    index = 0
                    his_data_list.pop(index)
                    his_label_list.pop(index)

                positive_label = p_nid.to(device)
                hard_negative_list = []
                with torch.no_grad():
                    if his_data is None \
                            or (global_step - 1) % opt.hard_negative_recompute_every_step == 0:
                        del his_data, his_data_label, his_data_out
                        torch.cuda.empty_cache()

                        his_data = torch.cat(tuple(his_data_list), 0).to(device)
                        his_data_label = torch.cat(tuple(his_label_list), 0).to(device)
                        his_data_out = speaker_net(his_data[:opt.batch_size])

                        while his_data_out.shape[0] < his_data.shape[0]:
                            s = his_data_out.shape[0]
                            tmp_out_i = speaker_net(his_data[s:s + opt.batch_size])
                            his_data_output = torch.cat((his_data_output, tmp_out_i), 0)
                            del tmp_out_i
                            torch.cuda.empty_cache()

                    anchor_output = speaker_net(anchor)
                    positive_output = speaker_net(positive)
                    distances_ap = torch.norm(positive_output - anchor_output, dim=1)

                    for a_index in range(anchor_output.shape[0]):
                        # distances_ap + opt.triplet_loss_margin - distances_an
                        distances_an = torch.norm(his_data_out - anchor_output[a_index, :], dim=1)
                        distances_an[his_data_label == positive_label[a_index]] = 1e30
                        distances_an[distances_an >= (distances_ap[a_index] + opt.triplet_loss_margin)] = 1e30

                        if opt.hard_negative_level == 1:
                            # hardest negatives
                            # distances_an[distances_an >= (distances_ap[a_index] + opt.triplet_loss_margin)] = 1e30
                            i1, i2 = distances_an.sort(0, descending=False)
                            hard_negative_index = i2.cpu().numpy()[:1][0]
                            index_list = list(np.where(distances_an.cpu().numpy() < 1e29)[0])
                            if len(index_list) > 0:
                                hard_negative_list.append(his_data[hard_negative_index, :].unsqueeze(0))
                            else:
                                hard_negative_list.append(negative[a_index, :].unsqueeze(0))
                        else:
                            # opt.hard_negative_level == 2
                            # semi-hard negatives, ap+margin>an>ap
                            distances_an[distances_an <= distances_ap[a_index]] = 1e30
                            distances_an[distances_an >= (distances_ap[a_index] + opt.triplet_loss_margin)] = 1e30

                            index_list = list(np.where(distances_an.cpu().numpy() < 1e29)[0])
                            if len(index_list) > 0:
                                hard_negative_list.append(his_data[random.choice(index_list), :].unsqueeze(0))
                            else:
                                hard_negative_list.append(negative[a_index, :].unsqueeze(0))

                del negative, hard_negative_list, anchor_output
                del positive_label, positive_output, distances_ap, distances_an
                torch.cuda.empty_cache()
                negative = torch.cat(tuple(hard_negative_list), 0)

            optimizer.zero_grad()
            a_output = speaker_net(anchor)
            p_output = speaker_net(positive)
            n_output = speaker_net(negative)
            loss = triplet_loss(a_output, p_output, n_output)
            loss.backward()
            optimizer.step()

            del anchor, positive, negative, a_output, p_output, n_output
            torch.cuda.empty_cache()

            avg_loss_meter.add(loss.item())
            # 写入tensor board 日志
            summary_writer.add_scalar('loss/b_loss', loss.item(), global_step)
            summary_writer.add_scalar('loss/avg_loss', avg_loss_meter.value()[0], global_step)
            summary_writer.add_scalar('net_params/lr', lr, global_step)

            if (global_step - 1) % opt.clustering_every_step == 0:
                identity_data = next(identity_speaker_data_gen).to(device)

                optimizer.zero_grad()
                identity_output = speaker_net(identity_data)
                i_loss = intra_class_loss(identity_output)
                i_loss.backward()
                optimizer.step()

                del identity_data, identity_output
                torch.cuda.empty_cache()
                summary_writer.add_scalar('loss/intra_class_loss', i_loss.item(), global_step)

            # 打印
            if (global_step - 1) % opt.print_every_step == 0:
                line_info = 'ep={:<3}({:<13}), '.format(epoch, str(i + 1) + '/' + str(total_batch))
                line_info += 'steps={:<9}, '.format(global_step)
                line_info += 'b_loss={:<3.2f}, avg_loss={:<3.2f}, '.format(loss.item(), avg_loss_meter.value()[0])
                print(line_info)

            # 定期保存
            if (global_step - 1) % total_batch + 1 in [int(x * total_batch) for x in opt.save_points_list]:
                date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
                save_path = os.path.join(fdir,
                                         'net_data/checkpoints/train/',
                                         '%s_%s_%s.pth' % (epoch, global_step, date_str))
                states_dict = {
                    'net': speaker_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch
                }
                torch.save(states_dict, save_path)

                # # 评估模型
                # eval_params = {
                #     'speaker_net': speaker_net, 'device': device, 'global_step': global_step,
                #     'ce_loss': ce_loss, 'summary_writer': summary_writer, 'train_dataset': train_dataset,
                # }
                # do_net_eval(**eval_params)

            # 保存last_checkpoint.pth以防意外情况导致训练进度丢失
            if (global_step - 1) % opt.last_checkpoint_save_interval == 0:
                save_path = os.path.join(fdir, 'net_data/checkpoints/train/', 'last_checkpoint.pth')
                states_dict = {
                    'net': speaker_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch
                }
                torch.save(states_dict, save_path)

            # 步骤加1
            global_step += 1

        epoch += 1
        scheduler.step(avg_loss_meter.value()[0])
        avg_loss_meter.reset()
        lr = optimizer.param_groups[0]['lr']

    summary_writer.close()


if __name__ == '__main__':
    fire.Fire()
