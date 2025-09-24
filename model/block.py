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

########################################
## S2Block
########################################

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y=None, **kwargs):
        if y is not None:
            return self.fn(x, y, **kwargs) + x
        else:
            return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y=None, **kwargs):
        if y is not None:
            return self.fn(self.norm(x), self.norm(y), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        # self.temperature = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.sa1 = nn.Linear(dim, inner_dim, bias=False)
        self.sa2 = nn.Linear(dim, inner_dim, bias=False)
        self.se1 = nn.Linear(dim, inner_dim, bias=False)
        self.se2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, y, mask=None):
        b, n, _, h = *x.shape, self.heads
        y1 = rearrange(self.sa1(y), 'b n (h d) -> b h n d', h=h)
        y2 = rearrange(self.sa2(y), 'b n (h d) -> b h n d', h=h)
        x1 = rearrange(self.se1(x), 'b n (h d) -> b h n d', h=h)
        x2 = rearrange(self.se2(x), 'b n (h d) -> b h n d', h=h)
        sacm = (y1 @ y2.transpose(-2, -1)) * self.scale
        secm = (x1.transpose(-2, -1) @ x2) * self.scale / (n/self.dim_head)  # b h d d
        sacm = sacm.softmax(dim=-1)
        secm = secm.softmax(dim=-1)
        out1 = torch.einsum('b h i j, b h j d -> b h i d', sacm, x1)
        out2 = torch.einsum('b h n i, b h i j -> b h n j', y1, secm)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out = out1 * out2
        out = self.to_out(out)
        return out


class S2Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, depth=1, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, MLP(dim, hidden_dim=mlp_dim, dropout=dropout)))]))

    def forward(self, x, y, mask=None):
        H = x.shape[2]
        x = rearrange(x, 'B C H W -> B (H W) C', H=H)
        y = rearrange(y, 'B C H W -> B (H W) C', H=H)
        for attn, ff in self.layers:
            x = attn(x, y, mask=mask)
            x = ff(x)
        x = rearrange(x, 'B (H W) C -> B C H W', H=H)
        return x

###################################################
## SCAM
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


class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3) # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  #B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) #B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r
    
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
                                          ResBlock(default_conv, n_feats, 3, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1),
                                        ResBlock(default_conv, n_feats, 3, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1))
        self.re_g = default_conv(n_feats, mc, 3)
        self.d1 = nn.Sequential(default_conv(mc, n_feats, 3),
                                ResBlock(default_conv, n_feats, 3, bias=True, bn=False,
                                act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1))

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

class LightGCMv2(nn.Module):
    '''
    FLOPs: 104.08 GFLOPs
    Params: 0.5173 M
    '''
    def __init__(self,n_feats, mc):
        super(LightGCMv2, self).__init__()
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

class GradExtractGCM(nn.Module):
    def __init__(self,n_feats, mc):
        super(GradExtractGCM, self).__init__()
        self.grad_msi = GradientExtractor()
        self.grad_pan = GradientExtractor()
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

########### WA ############

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops



class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        # 定义线性层，用于生成 Q、K、V
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        #Batch normalization layer
        self.OutBN = nn.BatchNorm2d(num_features=dim)

    def forward(self, x, context1=None, context2=None, mask=None):
        # 如果没有提供 context，则使用 x 作为 context（自注意力）
        # x shape: (batch_size, seq_len, dim)
        if context1 is None and context2 is None:
            context1 = x
            context2 = x

        batch_size, _, dim = x.size()      

        # x = x.view(x.shape[0], -1, x.shape[2]*x.shape[3])
        # context1 = context1.view(context1.shape[0], -1, context1.shape[2]*context1.shape[3])
        # context2 = context2.view(context2.shape[0], -1, context2.shape[2]*context2.shape[3])

        # # 转置输入，将维度变为 (batch_size, seq_len, dim)
        # x = x.transpose(1, 2)  # (batch_size, seq_len, dim)
        # context1 = context1.transpose(1, 2)  # (batch_size, context1_len, dim)
        # context2 = context2.transpose(1, 2)  # (batch_size, context2_len, dim)

        _, seq_len, _ = x.size()
        _, context1_len, _ = context1.size()
        _, context2_len, _ = context2.size()

        # 线性变换并拆分成多头
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)      # MSI
        k = self.k_linear(context1).view(batch_size, context1_len, self.num_heads, self.head_dim)    # Diff_MSI
        v = self.v_linear(context2).view(batch_size, context2_len, self.num_heads, self.head_dim)    # PAN

        # 调整维度以便计算
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, context_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, context_len, head_dim)

        # 计算注意力得分
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, context_len)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 计算注意力输出
        attn_output = torch.matmul(attn_probs, v)  # (batch_size, num_heads, seq_len, head_dim)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)  # (batch_size, seq_len, dim)

        # 输出线性层
        output = self.out_linear(attn_output) + x
        # output = self.OutBN(output)  # Batch normalization  (batch_size, seq_len, dim)

        # output = self.OutBN(output.transpose(1, 2).contiguous().view(batch_size, dim, int(math.sqrt(seq_len)), -1))     # unembedding
    
        return output