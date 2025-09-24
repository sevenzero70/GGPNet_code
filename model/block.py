import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import cv2

###################################################
## SSCAM
###################################################
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SSCAM(nn.Module):
    '''
    Spectral and Spetial Cross Attetion Module(SSCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_msi = LayerNorm2d(c)
        self.norm_pan = LayerNorm2d(c)
        self.msi_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.pan_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.msi_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.pan_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, msi, pan):
        Q_msi = self.msi_proj1(self.norm_msi(msi)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_pan = self.pan_proj1(self.norm_msi(msi)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_msi_T = self.msi_proj1(self.norm_msi(msi)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)
        Q_pan_T = self.pan_proj1(self.norm_pan(pan)).permute(0, 2, 1, 3) # B, H, c, W (transposed)

        V_msi = self.msi_proj2(msi).permute(0, 2, 3, 1)  # B, H, W, c
        V_pan = self.pan_proj2(pan).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        # cross attention matrix
        attention_mp = torch.matmul(Q_msi, Q_pan_T) * self.scale
        attention_pm = torch.matmul(Q_pan, Q_msi_T) * self.scale

        F_m2p = torch.matmul(torch.softmax(attention_mp, dim=-1), V_pan)    #B,H,W,c
        F_p2m = torch.matmul(torch.softmax(attention_pm, dim=-1), V_msi)    #B,H,W,c
        # scale 
        F_m2p = F_m2p.permute(0, 3, 1, 2) * self.beta
        F_p2m = F_p2m.permute(0, 3, 1, 2) * self.gamma
        return msi + F_m2p, pan + F_p2m

class SSCAMOne(nn.Module):
    '''
    Spectral and Spetial Cross Attetion Module(SSCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_msi = LayerNorm2d(c)
        self.norm_pan = LayerNorm2d(c)
        self.msi_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.pan_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.pan_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, msi, pan):
        Q_msi = self.msi_proj1(self.norm_msi(msi)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_pan_T = self.pan_proj1(self.norm_pan(pan)).permute(0, 2, 1, 3) # B, H, c, W (transposed)

        V_pan = self.pan_proj2(pan).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        # cross attention matrix
        attention_mp = torch.matmul(Q_msi, Q_pan_T) * self.scale

        F_m2p = torch.matmul(torch.softmax(attention_mp, dim=-1), V_pan)    #B,H,W,c
        # scale 
        F_m2p = F_m2p.permute(0, 3, 1, 2) * self.beta
        return msi + F_m2p

class SSCAMTwo(nn.Module):
    '''
    Spectral and Spetial Cross Attetion Module(SSCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_msi = LayerNorm2d(c)
        self.norm_pan = LayerNorm2d(c)
        self.msi_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.pan_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.msi_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.pan_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, msi, pan):
        Q_pan = self.pan_proj1(self.norm_msi(msi)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_msi_T = self.msi_proj1(self.norm_msi(msi)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_msi = self.msi_proj2(msi).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        # cross attention matrix
        attention_pm = torch.matmul(Q_pan, Q_msi_T) * self.scale

        F_p2m = torch.matmul(torch.softmax(attention_pm, dim=-1), V_msi)    #B,H,W,c
        # scale 
        F_p2m = F_p2m.permute(0, 3, 1, 2) * self.gamma
        return msi + F_p2m
################################################
## Gradient
################################################
class GradientExtractor(nn.Module):
    def __init__(self, method='log', kernel_size=5, sigma1=1.0, sigma2=2.0, canny_thresh=(100, 200)):
        """
        method: 'log' | 'dog' | 'canny'
        kernel_size: for Gaussian / Laplacian kernels
        sigma1, sigma2: used in DoG
        canny_thresh: tuple (low, high) for Canny
        """
        super(GradientExtractor, self).__init__()
        self.method = method.lower()
        self.kernel_size = kernel_size
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.canny_thresh = canny_thresh

    def forward(self, x):
        """
        x: [B, C, H, W] tensor, expected range [0, 1]
        return: gradient magnitude map, same shape as x
        """
        self.num_channels = x.shape[1]

        if self.method == 'log':
            return self._log(x)
        elif self.method == 'dog':
            return self._dog(x)
        elif self.method == 'canny':
            return self._canny(x)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _log(self, x):
        # LoG: Gaussian Blur -> Laplacian
        B, C, H, W = x.shape
        log_list = []
        for b in range(B):
            per_img = x[b].detach().cpu().numpy()
            per_log = []
            for c in range(C):
                I = per_img[c]
                I_blur = cv2.GaussianBlur(I, (self.kernel_size, self.kernel_size), self.sigma1)
                I_log = cv2.Laplacian(I_blur, ddepth=cv2.CV_32F)
                per_log.append(I_log)
            log_list.append(np.stack(per_log, axis=0))
        log_np = np.stack(log_list, axis=0)
        return torch.tensor(log_np, dtype=x.dtype, device=x.device)

    def _dog(self, x):
        """
        x: [B, C, H, W]
        return: DoG = Gaussian(sigma1) - Gaussian(sigma2)
        """
        kernel1 = self._create_gaussian_kernel(self.kernel_size, self.sigma1, x.device)
        kernel2 = self._create_gaussian_kernel(self.kernel_size, self.sigma2, x.device)
    
        padding = self.kernel_size // 2
        blur1 = F.conv2d(x, kernel1, padding=padding, groups=x.shape[1])
        blur2 = F.conv2d(x, kernel2, padding=padding, groups=x.shape[1])
        return blur1 - blur2
    
    def _create_gaussian_kernel(self, kernel_size, sigma, device):
        """
        Returns: Gaussian kernel for conv2d [C, 1, k, k]
        """
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.to(device=device, dtype=torch.float32)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        return kernel.repeat(self.num_channels, 1, 1, 1)  # [C, 1, k, k]

    def _canny(self, x):
        # Canny is non-differentiable; returns float32 in [0, 1]
        B, C, H, W = x.shape
        canny_list = []
        for b in range(B):
            per_img = x[b].detach().cpu().numpy()
            per_canny = []
            for c in range(C):
                I = (per_img[c] * 255).astype(np.uint8)
                edge = cv2.Canny(I, self.canny_thresh[0], self.canny_thresh[1])
                per_canny.append(edge.astype(np.float32) / 255.0)
            canny_list.append(np.stack(per_canny, axis=0))
        return torch.tensor(np.stack(canny_list), device=x.device, dtype=x.dtype)

class Get_gradient_nopadding_msi(nn.Module):
    def __init__(self, in_channels):
        super(Get_gradient_nopadding_msi, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()
        self.in_channels = in_channels

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)
        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        _x_ = x0

        for i in range(1, self.in_channels):
            x_ = x[:, i]
            x_v = F.conv2d(x_.unsqueeze(1), self.weight_v, padding=1)
            x_h = F.conv2d(x_.unsqueeze(1), self.weight_h, padding=1)

            x_ = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

            _x_ = torch.cat([_x_, x_], dim=1)
        
        return _x_

class Get_gradient_nopadding_d(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding_d, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]

        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        x = x0
        return x

class DenseProjection(nn.Module):
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        self.up = up
        if bottleneck:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)
            ])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        self.conv_1 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])
        self.conv_2 = nn.Sequential(*[
            projection_conv(nr, inter_channels, scale, not up),
            nn.PReLU(inter_channels)
        ])
        self.conv_3 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        a_0 = self.conv_1(x)
        b_0 = self.conv_2(a_0)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)

        out = a_0.add(a_1)
        return out

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {
        2: (6, 2, 2),
        4: (8, 4, 2),
        8: (12, 8, 2),
        16: (20, 16, 2)
    }[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d

    return conv_f(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding
    )

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class GCM(nn.Module):
    def __init__(self,n_feats, mc):
        super(GCM, self).__init__()
        self.grad_msi = Get_gradient_nopadding_msi(mc)
        self.grad_pan = Get_gradient_nopadding_d()
        self.upBlock = DenseProjection(mc, mc, 4, up=True, bottleneck=False)
        self.downBlock = DenseProjection(n_feats, n_feats, 4, up=False, bottleneck=False)
        self.c_pan = default_conv(1, n_feats, 3)
        self.c_msi = default_conv(mc, n_feats, 3)
        self.c_fuse = default_conv(n_feats,n_feats,3)

        self.rg_d = ResidualGroup(default_conv, n_feats, 3, reduction=16, n_resblocks=4)
        self.rb_rgbd = ResBlock(default_conv, n_feats, 3, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.fuse_process = nn.Sequential(nn.Conv2d(2*n_feats, n_feats, 1, 1, 0),
                                          ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4),
                                          ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4))
        # self.re_g = default_conv(n_feats,1,3)
        self.re_g = default_conv(n_feats, mc, 3)
        self.re_d = default_conv(n_feats, 1, 3)
        self.c_sab = default_conv(1, n_feats, 3)
        # self.sig = nn.Sigmoid()
        # self.d1 = nn.Sequential(default_conv(1,n_feats,3),
        #                         ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=8))
        self.d1 = nn.Sequential(default_conv(mc, n_feats, 3),
                                ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=8))

        self.CA = CALayer(n_feats,reduction=4)

        # grad_conv = [
        #     default_conv(1, n_feats, kernel_size=3, bias=True),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     default_conv(n_feats, n_feats, kernel_size=3, bias=True),
        # ]
        grad_conv = [
            default_conv(mc, n_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(n_feats, n_feats, kernel_size=3, bias=True),
        ]
        self.grad_conv = nn.Sequential(*grad_conv)
        self.grad_rg = nn.Sequential(ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4),
        ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4))

    def forward(self, lr_msi, pan):

        # lr_msi = self.upBlock(lr_msi)

        grad_pan = self.grad_pan(pan)
        out_grad_pan = grad_pan
        grad_d = self.grad_msi(lr_msi)
        out_grad_d = grad_d
        pan1 = self.c_pan(grad_pan)
        d1 = self.c_msi(grad_d)
        pan2 = self.rb_rgbd(pan1)
        d2 = self.rg_d(d1)
        cat1 = torch.cat([pan2,d2],dim=1)
        inn1 = self.fuse_process(cat1)
        d3 = d1 + self.CA(inn1)
        grad_d2 = self.c_fuse(d3)
        out_re = self.re_g(grad_d2) # channel=mc
        d4 = self.d1(lr_msi)
        grad_d3 = self.grad_conv(out_re) + d4
        grad_d4 = self.grad_rg(grad_d3) # channel=n_feats

        return out_re, grad_d4, out_grad_pan, out_grad_d

class LightGCM(nn.Module):
    '''
    FLOPs: 104.08 GFLOPs
    Params: 0.5173 M
    '''
    def __init__(self,n_feats, mc):
        super(LightGCM, self).__init__()
        self.grad_msi = Get_gradient_nopadding_msi(mc)
        self.grad_pan = Get_gradient_nopadding_d()
        # self.upBlock = DenseProjection(mc, mc, 4, up=True, bottleneck=False)
        # self.downBlock = DenseProjection(n_feats, n_feats, 4, up=False, bottleneck=False)
        self.c_pan = default_conv(1, n_feats, 3)
        self.c_msi = default_conv(mc, n_feats, 3)
        self.c_fuse = default_conv(n_feats,n_feats,3)

        self.rg_d = ResBlock(default_conv, n_feats, 3, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.rb_rgbd = ResBlock(default_conv, n_feats, 3, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1)
        self.fuse_process = nn.Sequential(nn.Conv2d(2*n_feats, n_feats, 1, 1, 0),
                                          ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=4))
        self.re_g = default_conv(n_feats, mc, 3)
        self.d1 = nn.Sequential(default_conv(mc, n_feats, 3),
                                ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=8))

        self.CA = CALayer(n_feats,reduction=4)

        grad_conv = [
            default_conv(mc, n_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(n_feats, n_feats, kernel_size=3, bias=True),
        ]
        self.grad_conv = nn.Sequential(*grad_conv)
        self.grad_rg = nn.Sequential(ResBlock(default_conv, n_feats, 3, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1))

    def forward(self, lr_msi, pan):

        # lr_msi = self.upBlock(lr_msi)

        grad_pan = self.grad_pan(pan)
        grad_d = self.grad_msi(lr_msi)
        out_grad_pan = grad_pan
        out_grad_d = grad_d

        pan1 = self.c_pan(grad_pan)
        d1 = self.c_msi(grad_d)
        pan2 = self.rb_rgbd(pan1)
        d2 = self.rg_d(d1)

        cat1 = torch.cat([pan2,d2],dim=1)
        inn1 = self.fuse_process(cat1)
        d3 = d1 + self.CA(inn1)
        grad_d2 = self.c_fuse(d3)
        out_re = self.re_g(grad_d2) # channel=mc

        d4 = self.d1(lr_msi)
        grad_d3 = self.grad_conv(out_re) + d4
        grad_d4 = self.grad_rg(grad_d3) # channel=n_feats

        return out_re, grad_d4, out_grad_pan, out_grad_d


