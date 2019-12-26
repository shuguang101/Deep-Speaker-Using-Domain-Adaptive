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

from torchnet import meter
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config import opt
from utils import common_util
from nets.speaker_net_cnn import SpeakerNetFC
from datasets.voxceleb1 import VoxCeleb1


def do_net_eval(**kwargs):
    speaker_net = kwargs['speaker_net']
    device = kwargs['device']
    global_step = kwargs['global_step']
    ce_loss = kwargs['ce_loss']
    summary_writer = kwargs['summary_writer']

    train_dataset = kwargs['train_dataset']
    test_dataset = kwargs['test_dataset']
    test_dataloader = kwargs['test_dataloader']

    # 清空缓存
    torch.cuda.empty_cache()
    # 切换到eval模式
    speaker_net.eval()
    # 度量
    avg_loss_meter = meter.AverageValueMeter()
    avg_acc_meter = meter.AverageValueMeter()
    with torch.no_grad():
        for i, (a, p, n, p_nid, p_did, n_nid, n_did) in enumerate(test_dataloader):
            # 测试集是训练集的子集, 会导致nid不一致
            raw_p_nid_list = p_nid.cpu().numpy().tolist()
            new_p_nid_list = []
            for raw_p_nid in raw_p_nid_list:
                raw_p_sid = test_dataset.nid2sid_dict[raw_p_nid]
                new_p_nid = train_dataset.sid2nid_dict[raw_p_sid]
                new_p_nid_list.append(new_p_nid)

            # 获取测试数据及标签
            positive = p.to(device)
            positive_label = torch.tensor(np.array(new_p_nid_list), dtype=torch.long).to(device)
            positive_out = speaker_net(positive)

            # loss
            loss = ce_loss(positive_out, positive_label)
            # acc
            _, predict_label = torch.max(positive_out, 1)
            correct_count = (predict_label == positive_label).sum()
            acc = correct_count.float() / positive_out.size(0)
            # avg meter
            avg_loss_meter.add(loss.item())
            avg_acc_meter.add(acc.item())
            # 清空缓存
            torch.cuda.empty_cache()
        # 写入日志
        summary_writer.add_scalar('eval/loss', avg_loss_meter.value()[0], global_step)
        summary_writer.add_scalar('eval/acc', avg_acc_meter.value()[0], global_step)

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

    # 获取参数
    opt_attrs = common_util.get_all_attribute(opt)
    params_dict = {k: getattr(opt, k) for k in opt_attrs}

    # 测试数据集参数
    dataset_train_param = {**params_dict, **{'dataset_type_name': 'train'}}
    dataset_test_param = {**params_dict, **{'dataset_type_name': 'test'}}

    # 读取训练数据
    # train_dataset, train_dataloader = common_util.load_data(opt, **dataset_train_param)

    # 读取测试数据
    test_dataset = VoxCeleb1(opt.test_used_dataset, **dataset_test_param)
    test_dataloader = DataLoader(test_dataset,
                                 shuffle=opt.shuffle,
                                 batch_size=opt.batch_size,
                                 num_workers=opt.num_workers,
                                 pin_memory=opt.pin_memory,
                                 timeout=opt.dataloader_timeout)

    train_dataset, train_dataloader = copy.deepcopy(test_dataset), copy.deepcopy(test_dataloader)

    # tensor board summary writer
    summary_writer = SummaryWriter(log_dir=os.path.join(fdir, 'summary_log_dir/pre_train/'))

    lr = opt.lr
    epoch, global_step = 1, 1
    device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')
    map_location = lambda storage, loc: storage.cuda(0) if opt.use_gpu else lambda storage, loc: storage

    speaker_net = SpeakerNetFC(train_dataset.num_of_speakers, device, opt.dropout_keep_prop)
    speaker_net.to(device)

    optimizer = torch.optim.SGD(speaker_net.parameters(),
                                lr=lr,
                                weight_decay=opt.weight_decay,
                                momentum=opt.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           patience=1,
                                                           factor=0.2)

    ce_loss = torch.nn.CrossEntropyLoss().to(device)

    if opt.pre_train_status_dict_path and os.path.exists(opt.pre_train_status_dict_path):
        print('load status dict \"%s\"' % opt.pre_train_status_dict_path)
        status_dict = torch.load(opt.status_dict_path, map_location)
        speaker_net.load_state_dict(status_dict['net'])
        optimizer.load_state_dict(status_dict['optimizer'])

        global_step = status_dict['global_step']
        epoch = status_dict['epoch']
        lr = status_dict['optimizer']['param_groups'][0]['lr']

    # 覆盖网络参数
    if opt.override_net_params:
        lr = opt.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        speaker_net.set_dropout_keep_prop(opt.dropout_keep_prop)

    avg_loss_meter = meter.AverageValueMeter()
    avg_acc_meter = meter.AverageValueMeter()

    while epoch <= opt.max_epoch:
        total_batch = train_dataloader.__len__()
        for i, (a, p, n, p_nid, p_did, n_nid, n_did) in enumerate(train_dataloader):
            # 清空cuda缓存
            torch.cuda.empty_cache()
            # run one step
            positive = p.to(device)
            positive_label = p_nid.to(device)

            # net backward
            optimizer.zero_grad()
            positive_out = speaker_net(positive)
            loss = ce_loss(positive_out, positive_label)
            loss.backward()
            optimizer.step()

            # 计算训练集上的准确率
            _, predict_label = torch.max(positive_out, 1)
            correct_count = (predict_label == positive_label).sum()
            acc = correct_count.float() / positive_out.size(0)
            # 统计平均值
            avg_loss_meter.add(loss.item())
            avg_acc_meter.add(acc.item())

            # 写入tensor board 日志
            summary_writer.add_scalar('loss/b_loss', loss.item(), global_step)
            summary_writer.add_scalar('loss/avg_loss', avg_loss_meter.value()[0], global_step)
            summary_writer.add_scalar('acc/b_acc', acc.item(), global_step)
            summary_writer.add_scalar('acc/avg_acc', avg_acc_meter.value()[0], global_step)
            summary_writer.add_scalar('net_params/lr', lr, global_step)
            summary_writer.add_scalar('mean/predicted_label', torch.mean(predict_label.float()).item(), global_step)
            summary_writer.add_scalar('mean/true_label', torch.mean(positive_label.float()).item(), global_step)

            # 打印
            if (global_step - 1) % opt.print_every_step == 0:
                line_info = 'ep={:<3}({:<13}), '.format(epoch, str(i + 1) + '/' + str(total_batch))
                line_info += 'steps={:<9}, '.format(global_step)
                line_info += 'b_loss={:<3.2f}, avg_loss={:<3.2f}, '.format(loss.item(), avg_loss_meter.value()[0])
                line_info += 'b_acc={:<2.2%}, avg_acc={:<2.2%}'.format(acc.item(), avg_acc_meter.value()[0])
                print(line_info)

            # 定期保存
            if global_step in [int(x * total_batch) for x in opt.save_points_list]:
                date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
                save_path = os.path.join(fdir, 'net_data/checkpoints/', '%s_%s_%s.pth' % (epoch, global_step, date_str))
                states_dict = {
                    'net': speaker_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch
                }
                torch.save(states_dict, save_path)

                # 评估模型
                eval_params = {
                    'speaker_net': speaker_net, 'device': device, 'global_step': global_step,
                    'ce_loss': ce_loss, 'summary_writer': summary_writer, 'train_dataset': train_dataset,
                    'test_dataset': test_dataset, 'test_dataloader': test_dataloader
                }
                do_net_eval(**eval_params)

            # 保存last_checkpoint.pth以防意外情况导致训练进度丢失
            if (global_step - 1) % opt.last_checkpoint_save_interval == 0:
                save_path = os.path.join(fdir, 'net_data/checkpoints/', 'last_checkpoint.pth')
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
        avg_acc_meter.reset()
        lr = optimizer.param_groups[0]['lr']

    summary_writer.close()


if __name__ == '__main__':
    fire.Fire()
