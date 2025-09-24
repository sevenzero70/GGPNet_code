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

# 梯度先和msi融合，然后加pan，现在用的是已经默认上采样的lms
class GGPNet(nn.Module):

    def __init__(self, model_opt=tuple, dim_head=16, se_ratio_mlp=0.5, se_ratio_rb=0.5):
        super(GGPNet, self).__init__()
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