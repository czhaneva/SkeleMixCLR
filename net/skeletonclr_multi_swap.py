import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class

import numpy as np
import random

trunk_ori_index = [4, 3, 21, 2, 1]
left_hand_ori_index = [9, 10, 11, 12, 24, 25]
right_hand_ori_index = [5, 6, 7, 8, 22, 23]
left_leg_ori_index = [17, 18, 19, 20]
right_leg_ori_index = [13, 14, 15, 16]

trunk = [i - 1 for i in trunk_ori_index]
left_hand = [i - 1 for i in left_hand_ori_index]
right_hand = [i - 1 for i in right_hand_ori_index]
left_leg = [i - 1 for i in left_leg_ori_index]
right_leg = [i - 1 for i in right_leg_ori_index]
body_parts = [trunk, left_hand, right_hand, left_leg, right_leg]


class SkeletonCLR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5, sigma=0.5, spa_l=1, spa_u=4, tem_l=1, tem_u=15,
                 repeat=1, swap_mode='swap', spatial_mode='semantic',
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'}, edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature
            self.sigma = sigma

            self.spa_l = spa_l
            self.spa_u = spa_u
            self.tem_l = tem_l
            self.tem_u = tem_u
            self.repeat = repeat
            self.swap_mode = swap_mode
            self.spatial_mode = spatial_mode

            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.BatchNorm1d(dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.BatchNorm1d(dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_label", torch.zeros(self.K) - torch.ones(self.K))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, label):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T
        self.queue_label[(ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = label

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    @torch.no_grad()
    def ske_swap(self, x):
        '''
        swap a batch skeleton
        T   64 --> 32 --> 16    # 8n
        S   25 --> 25 --> 25 (5 parts)
        '''
        N, C, T, V, M = x.size()
        tem_downsample_ratio = 4

        # generate swap swap idx
        idx = torch.arange(N)
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N

        # ------ Spatial ------ #
        if self.spatial_mode == 'semantic':
            Cs = random.randint(self.spa_l, self.spa_u)
            # sample the parts index
            parts_idx = random.sample(body_parts, Cs)
            # generate spa_idx
            spa_idx = []
            for part_idx in parts_idx:
                spa_idx += part_idx
            spa_idx.sort()
        elif self.spatial_mode == 'random':
            spa_num = random.randint(10, 15)
            spa_idx = random.sample(list(range(V)), spa_num)
            spa_idx.sort()
        else:
            raise ValueError('Not supported operation {}'.format(self.spatial_mode))
        # spa_idx = torch.tensor(spa_idx, dtype=torch.long).cuda()

        # ------ Temporal ------ #
        Ct = random.randint(self.tem_l, self.tem_u)
        tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
        rt = Ct * tem_downsample_ratio

        xst = x.clone()
        # begin swap
        if self.swap_mode == 'swap':
            xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                xst[randidx][:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :]
        elif self.swap_mode == 'zeros':
            xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = 0
        elif self.swap_mode == 'Gaussian':
            xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                torch.randn(N, C, rt, len(spa_idx), M).cuda()
        else:
            raise ValueError('Not supported operation {}'.format(self.swap_mode))
        # generate mask
        mask = torch.zeros(T // tem_downsample_ratio, V).cuda()
        mask[tem_idx:tem_idx + Ct, spa_idx] = 1

        return randidx, xst, mask.bool()

    def forward(self, im_q, im_k=None, label=None, topk=1, weight=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        if not self.pretrain:
            return self.encoder_q(im_q)

        randidx_list, im_pc_list, mask_list = [], [], []
        for i in range(self.repeat):
            randidx, im_pc, mask = self.ske_swap(im_q)
            randidx_list.append(randidx)
            im_pc_list.append(im_pc)
            mask_list.append(mask)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        p_list, c_list = [], []
        # compute swap
        for im_pc, mask in zip(im_pc_list, mask_list):
            p, c = self.encoder_q(im_pc, mask)
            p, c = F.normalize(p, dim=1), F.normalize(c, dim=1)
            p_list.append(p)
            c_list.append(c)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Loss instance
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        logits_reg_p_list = []
        # Loss region p
        for randidx, p, c in zip(randidx_list, p_list, c_list):
            l_pos = torch.einsum('nc,nc->n', [p, k[randidx]]).unsqueeze(-1)
            l_neg_1 = torch.einsum('nc,ck->nk', [p, self.queue.clone().detach()])
            l_neg_2 = torch.einsum('nc,nc->n', [p, c.clone().detach()]).unsqueeze(-1)
            logits_reg_p = torch.cat([l_pos, l_neg_1, l_neg_2], dim=1)
            # apply temperature
            logits_reg_p /= self.T
            logits_reg_p_list.append(logits_reg_p)

        # Loss region c
        logits_reg_c_list = []
        for c, p in zip(c_list, p_list):
            l_pos = torch.einsum('nc,nc->n', [c, k]).unsqueeze(-1)
            l_neg_1 = torch.einsum('nc,ck->nk', [c, self.queue.clone().detach()])
            l_neg_2 = torch.einsum('nc,nc->n', [c, p.clone().detach()]).unsqueeze(-1)
            logits_reg_c = torch.cat([l_pos, l_neg_1, l_neg_2], dim=1)
            # apply temperature
            logits_reg_c /= self.T
            logits_reg_c_list.append(logits_reg_c)

        # prob, topkidx = torch.topk(l_neg, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)
        topk_onehot_neg_1 = torch.zeros_like(l_neg_1)
        topk_onehot_neg_2 = torch.zeros_like(l_neg_2)
        # topk_onehot.scatter_(1, topkidx, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)
        pos_mask_reg_p = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot_neg_1, topk_onehot_neg_2],
                                  dim=1)
        pos_mask_reg_c = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot_neg_1, topk_onehot_neg_2],
                                  dim=1)
        target_mask = label.clone().unsqueeze(-1) == self.queue_label.clone().unsqueeze(0)


        # dequeue and enqueue
        self._dequeue_and_enqueue(k, label)

        return logits, pos_mask.detach(), logits_reg_p_list, pos_mask_reg_p.detach(), \
               logits_reg_c_list, pos_mask_reg_c.detach(), topk_onehot.detach(), target_mask.detach()
