# -*- coding:utf-8 -*-


if __name__ == '__main__':
    # use 'spawn' in main file's first line, to prevent deadlock occur
    import multiprocessing

    multiprocessing.set_start_method('spawn')

import fire
import torch
import os
import datetime
import numpy as np
import random
from functools import reduce
from torchnet import meter
from tensorboardX import SummaryWriter

from config import opt
from utils import common_util
from nets.speaker_net_cnn import SpeakerNetEM
from nets.discriminator_cnn import DANet
from nets.intra_class_loss import IntraClassLoss
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
                                                                                  max_people_to_compare=8,
                                                                                  max_bs=8,
                                                                                  min_bs=8)

    summary_writer.add_scalar('eval_eer/eer', eer, global_step)
    summary_writer.add_scalar('eval_eer/th', th_root, global_step)

    # 用于在tensor board中显示聚类效果
    em_mat, em_imgs, em_id = test_dataset.get_batch_speaker_data_with_icon(10, 50)
    em_mat_out = speaker_net(em_mat.to(device)).cpu()
    summary_writer.add_embedding(em_mat_out, metadata=em_id, label_img=em_imgs, global_step=global_step)

    # 测试完成切换回train模式
    speaker_net.train()


def train(**kwargs):
    # 获取该文件的目录
    fdir = os.path.split(os.path.abspath(__file__))[0]

    # 覆盖默认参数
    for k, v in kwargs.items():
        setattr(opt, k, v)
    num_features = opt.n_mels * (1+len(opt.used_delta_orders))

    # 获取参数
    opt_attrs = common_util.get_all_attribute(opt)
    params_dict = {k: getattr(opt, k) for k in opt_attrs}

    # 数据集参数
    dataset_train_param = {**params_dict, **{'dataset_type_name': 'train'}}
    dataset_test_param = {**params_dict, **{'dataset_type_name': 'test'}}

    # 读取训练数据
    train_dataset, train_dataloader = common_util.load_data(opt, **dataset_train_param)
    identity_speaker_data_gen = train_dataset.identity_speaker_data_gen_forever(opt.batch_size)
    test_dataset, test_dataloader = common_util.load_data(opt, **dataset_test_param)

    # # 读取测试数据
    # from datasets.voxceleb1 import VoxCeleb1
    # from datasets.voxceleb2 import VoxCeleb2
    # from torch.utils.data import DataLoader
    # import copy
    # train_dataset1 = VoxCeleb1(opt.test_used_dataset, **dataset_test_param)
    # train_dataset2 = VoxCeleb2(opt.test_used_dataset + '1', **dataset_test_param)
    # from datasets.merged_dataset import MergedDataset
    # train_dataset = MergedDataset(None, dataset_tuple=(train_dataset1,
    #                                                    train_dataset2), **dataset_test_param)
    # train_dataloader = DataLoader(train_dataset,
    #                               shuffle=opt.shuffle,
    #                               batch_size=opt.batch_size,
    #                               num_workers=opt.num_workers,
    #                               pin_memory=opt.pin_memory,
    #                               timeout=opt.dataloader_timeout)
    # identity_speaker_data_gen = train_dataset.identity_speaker_data_gen_forever(opt.batch_size)
    # test_dataset, test_dataloader = copy.deepcopy(train_dataset), copy.deepcopy(train_dataloader)

    # tensor board summary writer
    summary_writer = SummaryWriter(log_dir=os.path.join(fdir, 'net_data/summary_log_dir/train/'))

    lr = opt.lr
    epoch, global_step = 1, 1
    device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')
    map_location = lambda storage, loc: storage.cuda(0) if opt.use_gpu else lambda storage, loc: storage

    speaker_net = SpeakerNetEM(num_features, opt.dropout_keep_prop)
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

    optimizer = torch.optim.SGD(filter(lambda pa: pa.requires_grad, speaker_net.parameters()),
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
    raw_avg_loss_meter = meter.AverageValueMeter()
    da_avg_loss_meter = meter.AverageValueMeter()
    da_avg_acc_meter = meter.AverageValueMeter()

    # 判别器网络
    da_lr = opt.da_lr
    da_net = DANet(512, train_dataset.num_of_domain)
    da_net.to(device)
    da_optimizer = torch.optim.SGD(da_net.parameters(),
                                   lr=da_lr,
                                   weight_decay=opt.da_weight_decay,
                                   momentum=opt.da_momentum)
    da_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(da_optimizer,
                                                              mode='min',
                                                              patience=opt.da_patience,
                                                              factor=0.2)
    da_ce_loss = torch.nn.CrossEntropyLoss().to(device)

    if opt.pre_train_status_dict_path:
        print('load pre train model parameters')
        pre_trained_states_dict = torch.load(opt.pre_train_status_dict_path, map_location)
        model_state_dict = speaker_net.state_dict()
        part_of_pre_trained_states_dict = {k: v for k, v in pre_trained_states_dict['net'].items() if
                                           k in model_state_dict}
        model_state_dict.update(part_of_pre_trained_states_dict)
        speaker_net.load_state_dict(model_state_dict)
        da_net.load_state_dict(pre_trained_states_dict['da_net'])
        da_optimizer.load_state_dict(pre_trained_states_dict['da_optimizer'])
        da_lr = pre_trained_states_dict['da_optimizer']['param_groups'][0]['lr']

        del pre_trained_states_dict
        torch.cuda.empty_cache()

    if opt.status_dict_path:
        status_dict = torch.load(opt.status_dict_path, map_location)
        speaker_net.load_state_dict(status_dict['net'])
        da_net.load_state_dict(status_dict['da_net'])
        try:
            print('load status dict paramters')
            optimizer.load_state_dict(status_dict['optimizer'])
            da_optimizer.load_state_dict(status_dict['da_optimizer'])
        except Exception as e:
            print('warning: load optimizer status dict failed\r\n', e)

        global_step = status_dict['global_step'] + 1
        epoch = status_dict['epoch']
        lr = status_dict['optimizer']['param_groups'][0]['lr']
        da_lr = status_dict['da_optimizer']['param_groups'][0]['lr']

        del status_dict
        torch.cuda.empty_cache()

    # 覆盖网络参数
    if opt.override_net_params:
        lr = opt.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        da_lr = opt.da_lr
        for param_group in da_optimizer.param_groups:
            param_group['lr'] = da_lr
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
            positive_domain_id = p_did.to(device)
            negative_domain_id = n_did.to(device)

            # net backward
            # step 1: 训练判别器
            da_optimizer.zero_grad()
            speaker_net.eval()
            da_net.train()
            da_out = da_net(speaker_net(negative).detach())
            da_loss = da_ce_loss(da_out, negative_domain_id)
            da_loss.backward()
            da_optimizer.step()

            # 计算准确率
            _, predict_domain_id = torch.max(da_out, 1)
            da_correct_count = (negative_domain_id == predict_domain_id).sum()
            da_acc = da_correct_count.float() / da_out.size(0)

            da_avg_loss_meter.add(da_loss.item())
            da_avg_acc_meter.add(da_acc.item())
            # da_net
            summary_writer.add_scalar('da_net/b_loss', da_loss.item(), global_step)
            summary_writer.add_scalar('da_net/avg_loss', da_avg_loss_meter.value()[0], global_step)
            summary_writer.add_scalar('da_net/b_acc', da_acc.item(), global_step)
            summary_writer.add_scalar('da_net/avg_acc', da_avg_acc_meter.value()[0], global_step)
            summary_writer.add_scalar('da_net/lr', da_lr, global_step)

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
                            his_data_out = torch.cat((his_data_out, tmp_out_i), 0)
                            del tmp_out_i
                            torch.cuda.empty_cache()

                    anchor_out = speaker_net(anchor)
                    positive_out = speaker_net(positive)
                    distances_ap = torch.norm(positive_out - anchor_out, dim=1)

                    for a_index in range(anchor_out.shape[0]):
                        # distances_ap + opt.triplet_loss_margin - distances_an
                        distances_an = torch.norm(his_data_out - anchor_out[a_index, :], dim=1)
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

                del negative, anchor_out
                del positive_label, positive_out, distances_ap, distances_an
                negative = torch.cat(tuple(hard_negative_list), 0)
                del hard_negative_list
                torch.cuda.empty_cache()

            optimizer.zero_grad()
            da_net.eval()
            speaker_net.train()
            a_out = speaker_net(anchor)
            p_out = speaker_net(positive)
            n_out = speaker_net(negative)

            da_out_p = da_net(p_out)
            da_out_n = da_net(n_out)
            da_loss_p = da_ce_loss(da_out_p, positive_domain_id)
            da_loss_n = da_ce_loss(da_out_n, negative_domain_id)
            da_loss = da_loss_p + da_loss_n
            if da_avg_acc_meter.value()[0] >= opt.da_avg_acc_th and \
                    opt.da_every_step > 0 \
                    and (global_step - 1) % opt.da_every_step == 0:
                da_loss = 1.0 * da_loss
            else:
                da_loss = 0.0 * da_loss
            raw_loss = triplet_loss(a_out, p_out, n_out)
            loss = raw_loss - opt.da_lambda * da_loss
            loss.backward()
            optimizer.step()

            del anchor, positive, negative, a_out, p_out, n_out
            torch.cuda.empty_cache()

            avg_loss_meter.add(loss.item())
            raw_avg_loss_meter.add(raw_loss.item())
            # 写入tensor board 日志
            summary_writer.add_scalar('train/loss/raw_b_loss', raw_loss.item(), global_step)
            summary_writer.add_scalar('train/loss/b_loss', loss.item(), global_step)
            summary_writer.add_scalar('train/loss/avg_loss', avg_loss_meter.value()[0], global_step)
            summary_writer.add_scalar('train/loss/raw_avg_loss', raw_avg_loss_meter.value()[0], global_step)
            summary_writer.add_scalar('train/net_params/lr', lr, global_step)

            if opt.clustering_every_step > 0 and (global_step - 1) % opt.clustering_every_step == 0:
                identity_data = next(identity_speaker_data_gen).to(device)

                optimizer.zero_grad()
                identity_out = speaker_net(identity_data)
                i_loss = intra_class_loss(identity_out)
                i_loss.backward()
                optimizer.step()

                del identity_data, identity_out
                torch.cuda.empty_cache()
                summary_writer.add_scalar('train/loss/intra_class_loss', i_loss.item(), global_step)

            # 打印
            if (global_step - 1) % opt.print_every_step == 0:
                line_info = 'ep={:<3}({:<13}), '.format(epoch, str(i + 1) + '/' + str(total_batch))
                line_info += 'steps={:<9}, '.format(global_step)
                line_info += 'b_loss={:<3.2f}, avg_loss={:<3.2f}'.format(loss.item(), avg_loss_meter.value()[0])
                print(line_info)

            # 定期保存
            if (global_step - 1) % total_batch + 1 in [int(x * total_batch) for x in opt.save_points_list]:
                date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
                save_path = os.path.join(fdir,
                                         'net_data/checkpoints/train/',
                                         '%s_%s_%s.pth' % (epoch, global_step, date_str))
                states_dict = {
                    'net': speaker_net.state_dict(),
                    'da_net': da_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'da_optimizer': da_optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch
                }
                torch.save(states_dict, save_path)

                # 评估模型
                eval_params = {
                    'speaker_net': speaker_net, 'device': device, 'global_step': global_step,
                    'summary_writer': summary_writer, 'test_dataset': test_dataset,
                }
                do_net_eval(**eval_params)

            # 保存last_checkpoint.pth以防意外情况导致训练进度丢失
            if (global_step - 1) % opt.last_checkpoint_save_interval == 0:
                save_path = os.path.join(fdir, 'net_data/checkpoints/train/', 'last_checkpoint.pth')
                states_dict = {
                    'net': speaker_net.state_dict(),
                    'da_net': speaker_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'da_optimizer': da_optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch
                }
                torch.save(states_dict, save_path)

            # 步骤加1
            global_step += 1

        epoch += 1
        scheduler.step(avg_loss_meter.value()[0])
        da_scheduler.step(avg_loss_meter.value()[0])
        avg_loss_meter.reset()
        raw_avg_loss_meter.reset()
        da_avg_loss_meter.reset()
        da_avg_acc_meter.reset()
        lr = optimizer.param_groups[0]['lr']
        da_lr = da_optimizer.param_groups[0]['lr']

    summary_writer.close()


if __name__ == '__main__':
    fire.Fire()
