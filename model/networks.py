#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
from model.block import *
# from block import *


'''
将gradient、msi、pan进行作为qkv进行缝合
'''
class PReNetGradientMultiF(nn.Module):

    def __init__(self, model_opt=tuple, dim_head=16, se_ratio_mlp=0.5, se_ratio_rb=0.5):
        super(PReNetGradient, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )
        
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(self.msi_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(self.pan_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )

        # self.s2block = S2Block(self.kernel_c, self.kernel_c//dim_head, dim_head, int(self.kernel_c*se_ratio_mlp))
        self.ggblock = GCM(self.kernel_c, self.msi_c)
        self.fuse = SSCAM(self.kernel_c)
        self.attenfuse = MultiHeadAttention()



    def forward(self, input, pan_input):

        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []

        x = input
        y = pan_input
        y = self.raise_pan_dim(y)

        grad, grad_guided = self.ggblock(x, y)     # after gradiant fuse:grad->b,c,h,w(8,8,64,64) to calculate loss; grad_guided->(8,32,64,64)
        for i in range(self.iteration):
            fusion_x = self.attenfuse(self.raise_ms_dim(x), grad_guided, y)
            # x = self.conv0(torch.cat((self.raise_ms_dim(x), grad_guided), 1))
            # x = self.s2block(x, y)    # b,c,h,w:8,32,64,64
            # x, _ = self.fuse(x, y)
            # x = self.fuse(x, y)
            # x = self.conv0(torch.cat((x, y), 1))
            
            x = torch.cat((fusion_x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + pan_input
            x_list.append(x)
        x = x + input

        return x, x_list, grad, grad_guided

''' 将gradient和MSI做SSCAM 再加上PAN 在迭代块结束时加PAN 之前不加'''
class PReNetGradientFuse_(nn.Module):

    def __init__(self, model_opt=tuple, dim_head=16, se_ratio_mlp=0.5, se_ratio_rb=0.5):
        super(PReNetGradientFuse_, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )
        
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(self.msi_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(self.pan_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.ggblock = GCM(self.kernel_c, self.msi_c)
        self.fuse = SSCAM(self.kernel_c)



    def forward(self, input, pan_input):

        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []

        x = input
        y = pan_input
        y = self.raise_pan_dim(y)

        grad, grad_guided = self.ggblock(x, y)     # after gradiant fuse:grad->b,c,h,w(8,8,64,64) to calculate loss; grad_guided->(8,32,64,64)
        for i in range(self.iteration):
            x, _ = self.fuse(self.raise_ms_dim(x), grad_guided)
            # x = self.conv0(torch.cat((self.raise_ms_dim(x), grad_guided), 1))
            # x = self.s2block(x, y)    # b,c,h,w:8,32,64,64
            # x, _ = self.fuse(x, y)
            # x = self.fuse(x, y)
            
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + pan_input
            x_list.append(x)
        x = x + input

        return x, x_list, grad

''' 将gradient和MSI做SSCAM 再加上PAN '''
class PReNetGradientFuse(nn.Module):

    def __init__(self, model_opt=tuple, dim_head=16, se_ratio_mlp=0.5, se_ratio_rb=0.5):
        super(PReNetGradientFuse, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )
        
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(self.msi_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(self.pan_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.ggblock = GCM(self.kernel_c, self.msi_c)
        self.fuse = SSCAM(self.kernel_c)



    def forward(self, input, pan_input):

        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []

        x = input
        y = pan_input
        y = self.raise_pan_dim(y)

        grad, grad_guided = self.ggblock(x, y)     # after gradiant fuse:grad->b,c,h,w(8,8,64,64) to calculate loss; grad_guided->(8,32,64,64)
        for i in range(self.iteration):
            x, _ = self.fuse(self.raise_ms_dim(x), grad_guided)
            # x = self.conv0(torch.cat((self.raise_ms_dim(x), grad_guided), 1))
            # x = self.s2block(x, y)    # b,c,h,w:8,32,64,64
            # x, _ = self.fuse(x, y)
            # x = self.fuse(x, y)
            x = self.conv0(torch.cat((x, y), 1))
            
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + pan_input
            x_list.append(x)
        x = x + input

        return x, x_list, grad

# 梯度先和msi融合，然后加pan，现在用的是已经默认上采样的lms
class PReNetGradient(nn.Module):

    def __init__(self, model_opt=tuple, dim_head=16, se_ratio_mlp=0.5, se_ratio_rb=0.5):
        super(PReNetGradient, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        # self.conv_test = nn.Sequential(
        #     nn.Conv2d(self.kernel_c + self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
        #     nn.ReLU()
        #     )     # ablation test
        self.conv0 = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )
        
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(self.msi_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(self.pan_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.ggblock = GCM(self.kernel_c, self.msi_c)
        self.fuse = SSCAM(self.kernel_c)
        # self.fuse_cat = nn.Sequential(nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),nn.ReLU())



    def forward(self, input, pan_input):

        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []

        x = input
        y = pan_input
        y = self.raise_pan_dim(y)

        # out_y_feature = y
        # out_x_feature = self.raise_ms_dim(x)

        # grad, grad_guided, grad_pan, grad_msi = self.ggblock(x, y)     # after gradiant fuse:grad->b,c,h,w(8,8,64,64) to calculate loss; grad_guided->(8,32,64,64)
        grad, grad_guided, grad_pan, grad_msi = self.ggblock(x, pan_input)
        for i in range(self.iteration):
            # -----
            x = self.conv0(torch.cat((self.raise_ms_dim(x), grad_guided), 1))
            x, _ = self.fuse(x, y)
            # x = self.fuse_cat(torch.cat((x, y), 1))
            # x = x + y
            # -----
            # ----- 作为消融的baseline测试-0108_test_ablation_v2 this!!!
            # x = self.conv_test(torch.cat((self.raise_ms_dim(x), y, grad_guided), 1))

            # ----- 作为消融的baseline测试-0108_test_ablation
            # x = self.conv0(torch.cat((self.raise_ms_dim(x), grad_guided), 1))
            # x = self.conv_test(torch.cat((x, y, grad_guided), 1))

            x = self.conv0(torch.cat((x, y), 1))

            #-----------------------
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + pan_input
            x_list.append(x)
        x = x + input

        return x, x_list, grad
        # return x, x_list, grad, grad_guided, grad_pan, grad_msi

# ResNet/没有LSTM
class ResNetGradient(nn.Module):

    def __init__(self, model_opt=tuple, dim_head=16, se_ratio_mlp=0.5, se_ratio_rb=0.5):
        super(ResNetGradient, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        # self.conv_test = nn.Sequential(
        #     nn.Conv2d(self.kernel_c + self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
        #     nn.ReLU()
        #     )     # ablation test
        self.conv0 = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )

        self.conv = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )
        
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(self.msi_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(self.pan_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.ggblock = GCM(self.kernel_c, self.msi_c)
        self.fuse = SSCAM(self.kernel_c)
        # self.fuse_cat = nn.Sequential(nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),nn.ReLU())



    def forward(self, input, pan_input):

        x_list = []

        x = input
        y = pan_input
        y = self.raise_pan_dim(y)

        # out_y_feature = y
        # out_x_feature = self.raise_ms_dim(x)

        grad, grad_guided, grad_pan, grad_msi = self.ggblock(x, pan_input)     # after gradiant fuse:grad->b,c,h,w(8,8,64,64) to calculate loss; grad_guided->(8,32,64,64)
        for i in range(self.iteration):
            # -----
            x = self.conv0(torch.cat((self.raise_ms_dim(x), grad_guided), 1))
            x, _ = self.fuse(x, y)
            # x = self.fuse_cat(torch.cat((x, y), 1))
            # -----
            # ----- 作为消融的baseline测试-0108_test_ablation_v2 this!!!
            # x = self.conv_test(torch.cat((self.raise_ms_dim(x), y, grad_guided), 1))

            # ----- 作为消融的baseline测试-0108_test_ablation
            # x = self.conv0(torch.cat((self.raise_ms_dim(x), grad_guided), 1))
            # x = self.conv_test(torch.cat((x, y, grad_guided), 1))

            #-----------------------
            x = self.conv0(torch.cat((x, y), 1))


            # resblock
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + pan_input
            x_list.append(x)
        x = x + input

        return x, x_list, grad
        # return x, x_list, grad, grad_guided, grad_pan, grad_msi

# 不同的grad
class PReNetDiffGradient(nn.Module):

    def __init__(self, model_opt=tuple, dim_head=16, se_ratio_mlp=0.5, se_ratio_rb=0.5):
        super(PReNetDiffGradient, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        # self.conv_test = nn.Sequential(
        #     nn.Conv2d(self.kernel_c + self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
        #     nn.ReLU()
        #     )     # ablation test
        self.conv0 = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )
        
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(self.msi_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(self.pan_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.ggblock = GradExtractGCM(self.kernel_c, self.msi_c)
        self.fuse = SSCAM(self.kernel_c)

    def forward(self, input, pan_input):

        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []

        x = input
        y = pan_input
        y = self.raise_pan_dim(y)

        # out_y_feature = y
        # out_x_feature = self.raise_ms_dim(x)

        # grad, grad_guided, grad_pan, grad_msi = self.ggblock(x, y)     # after gradiant fuse:grad->b,c,h,w(8,8,64,64) to calculate loss; grad_guided->(8,32,64,64)
        grad, grad_guided, grad_pan, grad_msi = self.ggblock(x, pan_input)
        for i in range(self.iteration):
            # -----
            x = self.conv0(torch.cat((self.raise_ms_dim(x), grad_guided), 1))
            x, _ = self.fuse(x, y)

            x = self.conv0(torch.cat((x, y), 1))

            #-----------------------
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + pan_input
            x_list.append(x)
        x = x + input

        return x, x_list, grad
        # return x, x_list, grad, grad_guided, grad_pan, grad_msi

# 没有grad，没有fuse
class PReNetNoGradient(nn.Module):

    def __init__(self, model_opt=tuple, dim_head=16, se_ratio_mlp=0.5, se_ratio_rb=0.5):
        super(PReNetNoGradient, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        # self.conv_test = nn.Sequential(
        #     nn.Conv2d(self.kernel_c + self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
        #     nn.ReLU()
        #     )     # ablation test
        self.conv0 = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )
        
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(self.msi_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(self.pan_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )

    def forward(self, input, pan_input):

        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []

        x = input
        y = pan_input

        # out_y_feature = y
        # out_x_feature = self.raise_ms_dim(x)

        for i in range(self.iteration):

            x = self.conv0(torch.cat((self.raise_ms_dim(x), self.raise_pan_dim(y)), 1))

            #-----------------------
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + pan_input
            x_list.append(x)
        x = x + input

        return x, x_list
        # return x, x_list, grad, grad_guided, grad_pan, grad_msi

class PReNetSCAMTwo(nn.Module):
    def __init__(self, model_opt=tuple):
        super(PReNetSCAMTwo, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_ms = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )
        self.conv_pan = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.pan_c, 3, 1, 1),
            )
        
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(self.msi_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(self.pan_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        
        self.fuse = SCAM(self.kernel_c)
        # self.fuse = SSCAM(self.kernel_c)

        self.conv_final = nn.Sequential(
            nn.Conv2d(self.msi_c+self.pan_c, self.msi_c, 3, 1, 1), 
            nn.LeakyReLU()
        )

    def forward(self, input, pan_input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        h_ = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c_ = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []
        y_list = []

        x = input
        y = pan_input

        for i in range(self.iteration):

            x = self.raise_ms_dim(x)
            y = self.raise_pan_dim(y)
            x, y = self.fuse(x, y)

            #### x ####
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv_ms(x)

            x = x + input
            x_list.append(x)

            #### y ####
            y = torch.cat((y, h_), 1)
            i = self.conv_i(y)
            f = self.conv_f(y)
            g = self.conv_g(y)
            o = self.conv_o(y)
            c_ = f * c_ + i * g
            h_ = o * torch.tanh(c_)

            y = h_
            resy = y
            y = F.relu(self.res_conv1(y) + resy)
            resy = y
            y = F.relu(self.res_conv2(y) + resy)
            resy = y
            y = F.relu(self.res_conv3(y) + resy)
            resy = y
            y = F.relu(self.res_conv4(y) + resy)
            resy = y
            y = F.relu(self.res_conv5(y) + resy)
            y = self.conv_pan(y)

            y = y + pan_input
            y_list.append(y)

        x = self.conv_final(torch.cat((x, y), 1)) + input
        # x = x + y + input

        return x, x_list

class PReNetSCAMResTwo(nn.Module):
    def __init__(self, model_opt=tuple):
        super(PReNetSCAMResTwo, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_ms = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )
        self.conv_pan = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.pan_c, 3, 1, 1),
            )
        
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(self.msi_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(self.pan_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        
        self.fuse = SCAM(self.kernel_c)

        self.conv_final = nn.Sequential(
            nn.Conv2d(self.msi_c+self.pan_c, self.msi_c, 3, 1, 1), 
            nn.LeakyReLU()
        )

    def forward(self, input, pan_input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        h_ = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c_ = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []
        y_list = []

        x = input
        y = pan_input

        for i in range(self.iteration):

            x = self.raise_ms_dim(x)
            y = self.raise_pan_dim(y)
            x_fuse, y_fuse = self.fuse(x, y)

            # x = torch.cat((x, x_fuse), 1)
            # y = torch.cat((y, y_fuse), 1)
            x = x + x_fuse
            y = y + y_fuse

            #### x ####
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv_ms(x)

            x = x + input
            x_list.append(x)

            #### y ####
            y = torch.cat((y, h_), 1)
            i = self.conv_i(y)
            f = self.conv_f(y)
            g = self.conv_g(y)
            o = self.conv_o(y)
            c_ = f * c_ + i * g
            h_ = o * torch.tanh(c_)

            y = h_
            resy = y
            y = F.relu(self.res_conv1(y) + resy)
            resy = y
            y = F.relu(self.res_conv2(y) + resy)
            resy = y
            y = F.relu(self.res_conv3(y) + resy)
            resy = y
            y = F.relu(self.res_conv4(y) + resy)
            resy = y
            y = F.relu(self.res_conv5(y) + resy)
            y = self.conv_pan(y)

            y = y + pan_input
            y_list.append(y)

        x = self.conv_final(torch.cat((x, y), 1)) + input
        # x = x + y + input

        return x, x_list

class PReNetSCAMOne(nn.Module):
    """
    PReNet在concatenate部分换成SCAM
    """
    def __init__(self, model_opt=tuple):
        super(PReNetSCAMOne, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )
        
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(self.msi_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(self.pan_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        
        # self.fuse = SSCAM(self.kernel_c)

    def forward(self, input, pan_input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []

        x = input
        y = pan_input
        y = self.raise_pan_dim(y)

        for i in range(self.iteration):
            # x = torch.cat((input, x), 1)
            # x = torch.cat((pan, x), 1)
            # x = self.conv0(x)

            x = self.raise_ms_dim(x)
            # y = self.raise_pan_dim(y)
            # x, _ = self.fuse(x, y)
            x = self.conv0(torch.cat((x, y), 1))

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + pan_input
            # x = x + input
            x_list.append(x)

        return x, x_list

class WogradWSSCAM(nn.Module):
    """
    PReNet在concatenate部分换成SCAM
    """
    def __init__(self, model_opt=tuple):
        super(WogradWSSCAM, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )
        
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(self.msi_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(self.pan_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        
        self.fuse = SSCAM(self.kernel_c)

    def forward(self, input, pan_input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []

        x = input
        y = pan_input
        y = self.raise_pan_dim(y)

        for i in range(self.iteration):

            x = self.raise_ms_dim(x)
            # y = self.raise_pan_dim(y)
            x, _ = self.fuse(x, y)
            x = self.conv0(torch.cat((x, y), 1))

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + pan_input
            # x = x + input
            x_list.append(x)

        return x, x_list

class PReNetS2Block(nn.Module):
    """
    PReNet在concatenate部分换成S2Block
    """
    # def __init__(self, recurrent_iter=6, gpu_id=str, msi_c=int, pan_c=int):
    def __init__(self, model_opt=tuple, dim_head=16, se_ratio_mlp=0.5, se_ratio_rb=0.5):
        super(PReNetS2Block, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.msi_c+self.pan_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )
        
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(self.msi_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(self.pan_c, self.kernel_c, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.s2block = S2Block(self.kernel_c, self.kernel_c//dim_head, dim_head, int(self.kernel_c*se_ratio_mlp))



    def forward(self, input, pan_input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []

        x = input
        y = self.raise_pan_dim(pan_input)

        for i in range(self.iteration):

            x = self.raise_ms_dim(x)
            x = self.s2block(x, y)    # b,c,h,w:8,32,64,64

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + pan_input
            x_list.append(x)

        return x, x_list

class PReNet_MSI(nn.Module):
    def __init__(self, model_opt=tuple):
        super(PReNet_MSI, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.msi_c*2, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )

    def forward(self, input, pan):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list

class PReNet_PAN(nn.Module):
    def __init__(self, model_opt=tuple):
        super(PReNet_PAN, self).__init__()
        self.iteration  = model_opt.recurrent_iter
        # self.device     = 'cuda:'+ model_opt.gpu_id
        self.msi_c      = model_opt.msi_channel
        self.pan_c      = model_opt.pan_channel
        self.kernel_c   = model_opt.kernel_channel

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.msi_c+self.pan_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.msi_c, 3, 1, 1),
            )

    def forward(self, input, pan):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((pan, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + pan
            x_list.append(x)

        return x, x_list



class PReNet(nn.Module):
    def __init__(self, recurrent_iter=6, gpu_id=str):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter
        # self.device = gpu_id

        self.conv0 = nn.Sequential(
            nn.Conv2d(9, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(self.kernel_c + self.kernel_c, self.kernel_c, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(self.kernel_c, 8, 3, 1, 1),
            )

    def forward(self, input, pan):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, self.kernel_c, row, col)).cuda()

        x_list = []
        for i in range(self.iteration):
            # x = torch.cat((input, x), 1)
            x = torch.cat((pan, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + pan
            # x = x + input
            x_list.append(x)

        return x, x_list


class PReNet_LSTM(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_LSTM, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x1 = x
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x_list.append(x)

        return x, x_list


class PReNet_GRU(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_GRU, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_z = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        # self.conv_o = nn.Sequential(
        #     nn.Conv2d(32 + 32, 32, 3, 1, 1),
        #     nn.Sigmoid()
        #     )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x1 = torch.cat((x, h), 1)
            z = self.conv_z(x1)
            b = self.conv_b(x1)
            s = b * h
            s = torch.cat((s, x), 1)
            g = self.conv_g(s)
            h = (1 - z) * h + z * g

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)
            x_list.append(x)

        return x, x_list


class PReNet_x(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_x, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            #x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)
            x_list.append(x)

        return x, x_list


class PReNet_r(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet_r, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )


    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        #mask = Variable(torch.ones(batch_size, 3, row, col)).cuda()
        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            for j in range(5):
                resx = x
                x = F.relu(self.res_conv1(x) + resx)

            x = self.conv(x)
            x = input + x
            x_list.append(x)

        return x, x_list


## PRN
class PRN(nn.Module):
    def __init__(self, model_opt=tuple):
        super(PRN, self).__init__()
        self.iteration = model_opt.recurrent_iter
        # self.device = gpu_id

        self.conv0 = nn.Sequential(
            nn.Conv2d(9, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(64, 8, 3, 1, 1),
        )

    def forward(self, input, pan):

        x = input

        x_list = []
        for i in range(self.iteration):
            # x = torch.cat((input, x), 1)
            x = torch.cat((pan, x), 1)
            x = self.conv0(x)
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            # x = x + input
            x = x + pan
            x_list.append(x)

        return x, x_list


class PRN_r(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PRN_r, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )

        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

    def forward(self, input):

        x = input

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            for j in range(5):
                resx = x
                x = F.relu(self.res_conv1(x) + resx)

            x = self.conv(x)
            x = input + x
            x_list.append(x)

        return x, x_list
    
if __name__ == '__main__':

    from thop import profile
    import argparse

    parser = argparse.ArgumentParser(description="PReNet_train")
    parser.add_argument("--gpu_id", type=str, default='7', help='GPU id')
    parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages') # 6
    parser.add_argument("--msi_channel", type=int, default=4, help='MSI channel')
    parser.add_argument("--pan_channel", type=int, default=1, help='PAN channel')
    parser.add_argument("--kernel_channel", type=int, default=32, help='convolutional output behind the image input')   # 在服务器上计算时修改为64 light改为32
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    model = ResNetGradient(model_opt=opt).cuda()
    lms = torch.randn(1, 4, 256, 256).cuda()
    pan = torch.randn(1, 1, 256, 256).cuda()
    output = model(lms, pan)

    # 统计 FLOPs 和 Params
    flops, params = profile(model, inputs=(lms, pan), verbose=False)

    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.4f} M")