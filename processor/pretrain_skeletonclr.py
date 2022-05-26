#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor
from .knn_monitor import knn_predict
from tqdm import tqdm


class SkeletonCLR_Processor(PT_Processor):
    """
        Processor for SkeletonCLR Pretraining.
    """

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for [data1, data2, index], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            data1 = self.view_gen(data1)
            data2 = self.view_gen(data2)

            # forward
            output, target = self.model(data1, data2)
            if hasattr(self.model, 'module'):
                self.model.module.update_ptr(output.size(0))
            else:
                self.model.update_ptr(output.size(0))
            loss = self.loss(output, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.show_epoch_info()

    def view_gen(self, data):
        if self.arg.view == 'joint':
            pass
        elif self.arg.view == 'motion':
            motion = torch.zeros_like(data)

            motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

            data = motion
        elif self.arg.view == 'bone':
            Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                    (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

            bone = torch.zeros_like(data)

            for v1, v2 in Bone:
                bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
                bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]

            data = bone
        else:
            raise ValueError
        return data

    @torch.no_grad()
    def knn_monitor(self, epoch):
        self.model.encoder_q.eval()
        total_top1, total_top5, total_num, feature_bank, label_bank = 0.0, 0.0, 0, [], []
        with torch.no_grad():
            # generate feature bank
            for data, label in tqdm(self.data_loader['mem_train'], desc='Feature extracting'):
                data = data.float().to(self.dev, non_blocking=True)
                label = label.long().to(self.dev, non_blocking=True)

                data = self.view_gen(data)

                feature = self.model.encoder_q(data)
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
                label_bank.append(label)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            feature_labels = torch.cat(label_bank).to(feature_bank.device)
            # loop test data to predict the label by weighted knn search
            best_k, best_acc = 0, 0.
            for i in self.arg.knn_k:
                test_bar = tqdm(self.data_loader['mem_test'], desc='kNN-{}'.format(i))
                for data, label in test_bar:
                    data = data.float().to(self.dev, non_blocking=True)
                    label = label.float().to(self.dev, non_blocking=True)

                    data = self.view_gen(data)

                    feature = self.model.encoder_q(data)
                    feature = F.normalize(feature, dim=1)

                    pred_labels = knn_predict(feature, feature_bank, feature_labels, self.arg.knn_classes, i,
                                              self.arg.knn_t)

                    total_num += data.size(0)
                    total_top1 += (pred_labels[:, 0] == label).float().sum().item()
                    test_bar.set_postfix({'k': i, 'Accuracy': total_top1 / total_num * 100})
                acc = total_top1 / total_num * 100
                if acc > best_acc:
                    best_k, best_acc = i, acc
                self.train_writer.add_scalar('KNN-{}'.format(i), acc, epoch)
        if epoch in self.arg.KNN_show:
            try:
                self.KNN_epoch_results[epoch] = acc if acc > self.knn_best_result else self.knn_best_result
            except:
                self.KNN_epoch_results = {}
                self.KNN_epoch_results[epoch] = acc if acc > self.knn_best_result else self.knn_best_result
        self.knn_current_best_k = best_k
        self.knn_current_result = best_acc

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--cos', type=int, default=0, help='use cosine lr schedule')
#        parser.add_argument('--knn_k', type=int, default=[], nargs='+', help='KNN-K')
#        parser.add_argument('--knn_classes', type=int, default=60, help='use cosine lr schedule')
#        parser.add_argument('--knn_t', type=int, default=0.1, help='use cosine lr schedule')
#        parser.add_argument('--KNN_show', type=int, default=[], nargs='+',
#                            help='the epoch to show the best KNN result')
        # endregion yapf: enable

        return parser
