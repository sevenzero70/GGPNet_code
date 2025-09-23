##
## This is useful to test small size model: PReNet, PReSCAM, PReSSCAM, PReGrad
##

# import sys
# import os
# directory_path = os.path.abspath('/data3/lianglanyue/PReNet_PAN1/sota_models/BDT/model_SR_x4.py')
# sys.path.append(directory_path)

import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from model.networks import *
from model.block import *
import time 
from load_train_data import Dataset_Pro
from torch.utils.data import DataLoader
import re
from SSIM import SSIM
from metrics import *
from PIL import Image
import scipy.io as sio

from sota_models.BiMPan.BiMPan import *
from sota_models.DCINN.dcinn_ps import *
from sota_models.TDNet.model_8band import tdnet
from model.model_SR_x4 import *
from sota_models.U2Net.u2net import *
from sota_models.LGPConv.model import *
from sota_models.LAGConv.model import *
from sota_models.CML.model import *
from sota_models.GPPNN.GPPNN import *
# from sota_models.canconv.models.cannet import *
from sota_models.UTeRM.UTeRM_CNN import *
from sota_models.WFANet.net_torch import *

import wandb
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from thop import profile

def model_type(model_name):
    ########## Test/Ablation ##########
    if model_name == "PReNet_MSI":
        return PReNet_MSI
    elif model_name == "PReNet_PAN":
        return PReNet_PAN
    elif model_name == "PRN":
        return PRN
    elif model_name == "PReNetSCAMOne":
        return PReNetSCAMOne
    elif model_name == "PReNetSCAMResTwo":
        return PReNetSCAMResTwo
    elif model_name == "PReNetSCAMTwo":
        return PReNetSCAMTwo
    elif model_name == "PReNetS2Block":
        return PReNetS2Block
    # elif model_name == "PReNetHyperTrans":
    #     return PReNetHyperTrans
    elif model_name == "PReNetGradientFuse":
        return PReNetGradientFuse
    elif model_name == "PReNetGradientFuse_":
        return PReNetGradientFuse_
    ########## Proposed Method ##########
    elif model_name == "PReNetGradient":
        return PReNetGradient
    elif model_name == "ResNetGradient":
        return ResNetGradient
    elif model_name == "PReNetDiffGradient":
        return PReNetDiffGradient
    elif model_name == "PReNetNoGradient":
        return PReNetNoGradient
    ########## Comparison ##########
    elif model_name == "BiMPan":
        return BiMPan
    elif model_name == "DCINN":
        return DCINN
    elif model_name == "TDNet":
        return tdnet
    elif model_name == "BDT":
        return Bidinet
    elif model_name == "U2Net":
        return U2Net        # U2Net太大了，要用test_l
    elif model_name == "LGPConv":
        return NET
    elif model_name == "LAGConv":
        return LACNET
    elif model_name == "CML":
        return CMLNet
    elif model_name == "GPPNN":
        return GPPNN
    elif model_name == "WogradWSSCAM":
        return WogradWSSCAM
    # elif model_name == "canconv":
    #     return CANNet
    elif model_name == "UTeRM":
        return LRTC_Net
    elif model_name == "WFANet":
        return HWViT
    else:
        raise ValueError(f"Unknown model name: {model_name}")

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="/data3/lianglanyue/PReNet_PAN1/logs/0702/lightgcmv2_gf2", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="../Datasets/GF2_Deng/test_h5/test_multiExm1.h5", help='path to training data')
parser.add_argument("--save_path", type=str, default="results/NCrebuttal/0715/gf2_light_test", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument("--model_name", type=model_type, default='PReNetGradient', help='choose model')
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--msi_channel", type=int, default=4, help='MSI channel')
parser.add_argument("--pan_channel", type=int, default=1, help='PAN channel')
parser.add_argument("--kernel_channel", type=int, default=64, help='convolutional output behind the image input')   # 在服务器上计算时修改为64
parser.add_argument("--data_type", type=str, default='PReNetGradient', help='reduce, full')
parser.add_argument("--img_save_path", type=str, default='RGB_img')
parser.add_argument("--value_txt", type=str, default='value_info')
parser.add_argument("--vmax", type=float , default=0.1, help='diff max value')
# Upsample
parser.add_argument("--up_scale", type=int, default=1, help='if up_scale=1, MSI input id=lms; else MSI input id=ms')
parser.add_argument("--up_way", type=str, default='bicubic', help='pixel shuffle; bicubic; bilinear')

opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# 可视化
import os
import matplotlib.pyplot as plt

def calculate_channel_correlation(image):
    """
    计算多通道图像中两两通道的相关性（Pearson Correlation Coefficient）。

    Args:
        image (numpy.ndarray): 形状为 [C, H, W] 的多通道图像。

    Returns:
        numpy.ndarray: 相关性矩阵，形状为 [C, C]。
    """
    # 提取通道数和展平每个通道
    C, H, W = image.shape
    flattened = image.reshape(C, -1)  # [C, H*W]

    # 初始化相关性矩阵
    correlation_matrix = np.zeros((C, C))

    # 计算两两通道的相关性
    for i in range(C):
        for j in range(C):
            # 提取两个通道
            channel_i = flattened[i]
            channel_j = flattened[j]
            # 中心化
            mean_i = np.mean(channel_i)
            mean_j = np.mean(channel_j)
            centered_i = channel_i - mean_i
            centered_j = channel_j - mean_j
            # 计算相关性
            numerator = np.sum(centered_i * centered_j)
            denominator = np.sqrt(np.sum(centered_i ** 2) * np.sum(centered_j ** 2))
            correlation_matrix[i, j] = numerator / denominator

    return correlation_matrix

def plot_grouped_correlation(channels=4,
                             img_cor=None, grad_cor=None, 
                             title='img_grad_correlation', 
                             xlabel='channels', 
                             ylabel='correlation', 
                             colors=('skyblue', 'salmon'), 
                             ylim=(0, 1), 
                             save_path=None, 
                             figsize=(6, 4)):
    """
    绘制两个相关性值集的分组柱状图。

    :param channels: List[str]，通道标签列表，如 ['通道1', '通道2', ...]
    :param img_cor: Tensor/List[float]，第一组相关性值，如 img_cor
    :param grad_cor: Tensor/List[float]，第二组相关性值，如 grad_cor
    :param title: str，图表标题（默认 '相关性对比'）
    :param xlabel: str，X轴标签（默认 '通道'）
    :param ylabel: str，Y轴标签（默认 '相关性'）
    :param colors: Tuple[str, str]，两组柱子的颜色（默认 ('skyblue', 'salmon')）
    :param ylim: Tuple[float, float]，Y轴范围（默认 (-1, 1)）
    :param save_path: str 或 None，若提供路径则保存图像，否则仅显示（默认 None）
    :param figsize: Tuple[int, int]，图像大小（默认 (10, 6)）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 检查输入长度是否一致
    if not (channels == len(img_cor) == len(grad_cor)):
        raise ValueError("`channels`, `img_cor` 和 `grad_cor` 的长度必须一致。")
    
    # 设置柱状图的位置
    x = np.arange(channels)  # 通道的位置
    width = 0.15  # 柱子的宽度
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制 img_cor 柱子
    bars1 = ax.bar(x - width/2, img_cor, width, label='img_cor', color=colors[0])
    
    # 绘制 grad_cor 柱子
    bars2 = ax.bar(x + width/2, grad_cor, width, label='grad_cor', color=colors[1])
    
    # 添加标签和标题
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}' for c in range(channels)])
    ax.set_ylim(ylim)  # 设置Y轴范围
    
    # 添加网格线（可选）
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    
    # 添加图例
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 将图例移到右上方的图像外部
    
    # 在柱子上方添加数值标签
    def autolabel(bars):
        """在每个柱子上方添加数值标签"""
        for bar in bars:
            height = bar.get_height()
            if height >= 0:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
            else:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, -15),  # 15 points vertical offset
                            textcoords="offset points",
                            ha='center', va='top', fontsize=8, fontweight='bold')
    
    autolabel(bars1)
    autolabel(bars2)
    
    # 调整布局以防止标签被截断
    fig.tight_layout()
    
    # 保存图形（可选）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()


def visualize_feature_maps(out_x_fea_vis, save_path, cols=8, name='out_x_fea_channels'):
    """
    可视化特征图并保存为图片

    参数:
    out_x_fea (torch.Tensor): 输入的特征图，形状为 [C, H, W]
    cols (int): 每行显示的子图数量，默认为8
    save_path (str): 保存图片的路径，默认为 'out_x_fea_channels.png'
    """
    
    # 计算需要显示的特征图数量，最多显示64个通道
    num_channels = min(out_x_fea_vis.shape[0], 64)
    
    # 计算子图的行数
    rows = (num_channels + cols - 1) // cols
    
    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 2.5))
    axes = axes.flatten()
    
    # 遍历每个通道并绘制特征图
    for i in range(num_channels):
        out_x_fea_map = out_x_fea_vis[i, :, :]
        ax = axes[i]
        im = ax.imshow(out_x_fea_map, cmap='viridis')
        ax.set_title(f'{name} Ch{i}')
        ax.axis('off')
    
    # 关闭多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # 调整布局并保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, name+'.png'))
    plt.close()

# 示例调用
# visualize_feature_maps(out_x_fea, cols=8, save_path=os.path.join(grad_path, 'out_x_fea_channels.png'))


# 相关性计算
def correlation_highdim_single(x, y):
    """
    Calculate Pearson correlation between a single-channel numpy `x` and a multi-channel numpy `y`.

    Args:
        x (numpy): numpy of shape [1, H, W].
        y (numpy): numpy of shape [C, H, W].

    Returns:
        numpy: Correlation values of shape [C], one value per channel.
    """
    C, H, W = y.shape

    # Flatten spatial dimensions
    x_flat = x.reshape(x.shape[0], -1)  # [1, H*W]
    y_flat = y.reshape(y.shape[0], -1)  # [C, H*W]

    # Center the data by subtracting the mean
    x_mean = np.mean(x_flat, axis=1, keepdims=True)  # [1, 1]
    y_mean = np.mean(y_flat, axis=1, keepdims=True)  # [C, 1]
    x_centered = x_flat - x_mean  # [1, H*W]
    y_centered = y_flat - y_mean  # [C, H*W]

    # Compute covariance
    cov = np.sum(x_centered * y_centered, axis=1) / (H * W - 1)  # [C]

    # Compute variances
    var_x = np.sum(x_centered ** 2, axis=1) / (H * W - 1)  # [1]
    var_y = np.sum(y_centered ** 2, axis=1) / (H * W - 1)  # [C]

    # Compute Pearson correlation
    correlation = cov / np.sqrt(var_x * var_y)  # [C]
    return correlation

def visual_grad():

    rgb_path_base   = os.path.join(opt.save_path, opt.img_save_path)
    data_pca        = os.path.join(opt.save_path, 'PCA')
    data_histogram  = os.path.join(opt.save_path, 'histogram')
    data_diff       = os.path.join(opt.save_path, 'diff')
    mat_base        = os.path.join(opt.save_path, 'mat')


    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # tensorboard
    # writer = SummaryWriter(log_dir=opt.save_path)
    # wandb.init(project="Grad_map", name=opt.save_path, config=opt)

    
    # Build model
    print('Loading model ...\n')
    model = opt.model_name(model_opt=opt)
    print_network(model)
    # grad_net = Get_gradient_nopadding_msi(opt.msi_channel)
    if opt.use_GPU:
        model = model.cuda()

    # # 注册钩子函数
    # feature_maps = []
    # def hook_fn(module, input, output):
    #     feature_maps.append((module.__class__.__name__, output))
    # def register_hooks(model):
    #     for name, layer in model.named_modules():
    #         if isinstance(layer, nn.Conv2d):
    #             layer.register_forward_hook(hook_fn)
    # register_hooks(model)
    
    t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
    pth_list.append(os.path.join(opt.logdir, 'net_latest.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        pca_path        = os.path.join(data_pca, os.path.splitext(os.path.basename(pth))[0])
        histogram_path  = os.path.join(data_histogram, os.path.splitext(os.path.basename(pth))[0])
        diff_path       = os.path.join(data_diff, os.path.splitext(os.path.basename(pth))[0])
        mat_path        = os.path.join(mat_base, os.path.splitext(os.path.basename(pth))[0])
        
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(pca_path, exist_ok=True)
        os.makedirs(histogram_path, exist_ok=True)
        os.makedirs(diff_path, exist_ok=True)
        os.makedirs(mat_path, exist_ok=True)

        print(pth)
        model.load_state_dict(torch.load(pth))
        model.eval()

        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []
        img_cor_list = []
        grad_cor_list = []

        with torch.no_grad():
            for iteration, batch in enumerate(loader_valid, 1):
                # feature_maps = []  # 重置特征图列表

                if opt.up_scale == 1:
                    gt, pan, lms = batch[0].cuda(), batch[3].cuda(), batch[1].cuda()

                else:
                    gt, pan, ms = batch[0].cuda(), batch[3].cuda(), batch[4].cuda()
                    # pixel shuffle
                    if opt.up_way == 'pixel_shuffle':
                        lms = F.pixel_shuffle(ms, upscale_factor=opt.up_scale)
                    # bicubic
                    elif opt.up_way == 'bicubic':
                        lms = F.interpolate(ms, scale_factor=opt.up_scale, mode='bicubic')
                    # bilinear
                    elif opt.up_way == 'bilinear':
                        lms = F.interpolate(ms, scale_factor=opt.up_scale, mode='bilinear')

                # gt, pan, lms = batch[0].cuda(), batch[3].cuda(), batch[1].cuda()
                sr, _, grad, grad_guided, out_x_fea, out_y_fea = model(lms, pan)    # grad  
                # sr, _ = model(lms, pan)         # w/o grad
                # metric
                gt = gt.detach().cpu().numpy()
                sr = sr.detach().cpu().numpy()
                lms = lms.detach().cpu().numpy()
                pan = pan.detach().cpu().numpy()
                out_x_fea = out_x_fea.detach().cpu().numpy()
                out_y_fea = out_y_fea.detach().cpu().numpy()

                ergas   = calc_ergas(gt, sr)
                psnr    = calc_psnr(gt, sr)
                rmse    = calc_rmse(gt, sr)
                sam     = calc_sam(gt, sr)
                scc     = calc_scc(gt, sr)
                q2n     = calc_qindex(gt, sr)
                epoch_val_ergas.append(ergas)
                epoch_val_psnr.append(psnr)
                epoch_val_rmse.append(rmse)
                epoch_val_sam.append(sam)
                epoch_val_scc.append(scc)
                epoch_val_q2n.append(q2n)
                # # save mat
                # savemat('{}.mat'.format(iteration), sr)
                # save rgb
                gt = np.squeeze(gt, 0)
                sr = np.squeeze(sr, 0)
                grad = np.squeeze(grad, 0)
                grad_guided = np.squeeze(grad_guided, 0)
                out_x_fea = np.squeeze(out_x_fea, 0)
                out_y_fea = np.squeeze(out_y_fea, 0)
                lms = np.squeeze(lms, 0)
                # diff = diff_gt_sr(gt, sr)

                pca_analys(sr, os.path.join(pca_path, str(iteration)))
                histogram_analys(sr, os.path.join(histogram_path, str(iteration)))
                img_difference(gt, sr, os.path.join(diff_path, str(iteration)), vmin=0, vmax=opt.vmax)

                gt_img = save_img(gt, opt.msi_channel)
                sr_img = save_img(sr, opt.msi_channel)
                # diff_img = save_img(diff)
                gt_img_save = Image.fromarray(gt_img.astype(np.uint8), 'RGB')
                sr_img_save = Image.fromarray(sr_img.astype(np.uint8), 'RGB')
                # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
                gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
                sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))
                mat_name = os.path.join(mat_path, "output_mulExm_" + str(iteration) + ".mat")
                sio.savemat(mat_name, {'sr': sr})
                # diff_img_save.save('{}/diff_{}.png'.format(rgb_path, iteration))

            
                # 创建保存grad和grad_guided的目录
                grad_path = os.path.join(opt.save_path, 'gradients', os.path.splitext(os.path.basename(pth))[0], str(iteration))
                os.makedirs(grad_path, exist_ok=True)

                grad_vis = grad.detach().cpu()
                guided_vis = grad_guided.detach().cpu()
                
                # img_o_cor = calculate_channel_correlation(lms)
                # grad_o_cor = calculate_channel_correlation(out_y_fea)
                # img_channels_cor = correlation_highdim_single(np.expand_dims(lms[0], axis=0), np.expand_dims(lms[1], axis=0))
                # grad_channels_cor = correlation_highdim_single(np.expand_dims(out_y_fea[0], axis=0), np.expand_dims(out_y_fea[1], axis=0))
                img_cor = correlation_highdim_single(pan, lms)
                grad_cor = correlation_highdim_single(out_x_fea, out_y_fea)
                img_cor_list.append(img_cor)
                grad_cor_list.append(grad_cor)

                # 绘制grad和grad_guided的柱状图
                plot_grouped_correlation(channels=opt.msi_channel, img_cor=img_cor, grad_cor=grad_cor, save_path=os.path.join(grad_path, 'correlation.png'))

                # 绘制特征图
                visualize_feature_maps(grad_vis, grad_path, cols=8, name='grad_channels')
                visualize_feature_maps(guided_vis, grad_path, cols=8, name='guided_grad_channels')
                visualize_feature_maps(out_x_fea, grad_path, cols=8, name='out_x_fea_channels')
                visualize_feature_maps(out_y_fea, grad_path, cols=8, name='out_y_fea_channels')

                # 使用wandb记录特征图
                # wandb.log({
                #     f'Gradients/Original_{iteration}': wandb.Image(os.path.join(grad_path, 'grad_channels.png')),
                #     f'Gradients/Guided_{iteration}': wandb.Image(os.path.join(grad_path, 'guided_grad_channels.png')),
                #     f'Gradients/Out_x_fea_{iteration}': wandb.Image(os.path.join(grad_path, 'out_x_fea_channels.png')),
                #     f'Gradients/Out_y_fea_{iteration}': wandb.Image(os.path.join(grad_path, 'out_y_fea_channels.png')),
                #     F'Gradients/Correlation_{iteration}': wandb.Image(os.path.join(grad_path, 'correlation.png'))
                # })
            
            plot_grouped_correlation(channels=opt.msi_channel, img_cor=np.mean(img_cor_list, axis=0), grad_cor=np.mean(grad_cor_list, axis=0), save_path=os.path.join(opt.save_path, 'correlation_avg.png'))

        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_ergas = np.nanmean(np.array(epoch_val_ergas))
        v_psnr  = np.nanmean(np.array(epoch_val_psnr))
        v_rmse  = np.nanmean(np.array(epoch_val_rmse))
        v_sam   = np.nanmean(np.array(epoch_val_sam))
        v_scc   = np.nanmean(np.array(epoch_val_scc))
        v_q2n   = np.nanmean(np.array(epoch_val_q2n))
        print('[pth:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}'.format(pth, v_ergas, v_psnr, v_rmse, v_sam, v_scc, v_q2n))

        with open('{}/{}.txt'.format(opt.save_path, opt.value_txt), 'a') as f:
            f.write(f'[pth:{pth}], SAM={v_sam:.4f}, RMSE={v_rmse:.4f}, ERGAS={v_ergas:.4f}, PSNR={v_psnr:.4f}, SCC={v_scc:.4f}, Q2n={v_q2n:.4f}, time={total_time:.4f}\n')
        t_end   = time.time()
        print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        t_start = time.time()
    
    # writer.close()
    # 结束 wandb 运行
    wandb.finish()


def ablation_main():

    rgb_path_base   = os.path.join(opt.save_path, opt.img_save_path)
    data_pca        = os.path.join(opt.save_path, 'PCA')
    data_histogram  = os.path.join(opt.save_path, 'histogram')
    data_diff       = os.path.join(opt.save_path, 'diff')
    mat_base        = os.path.join(opt.save_path, 'mat')

    # os.makedirs(rgb_path_base, exist_ok=True)
    # os.makedirs(data_pca, exist_ok=True)
    # os.makedirs(data_histogram, exist_ok=True)


    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # Build model
    print('Loading model ...\n')
    model = opt.model_name(model_opt=opt)
    print_network(model)
    # grad_net = Get_gradient_nopadding_msi(opt.msi_channel)
    if opt.use_GPU:
        model = model.cuda()
    
    t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'best*'))
    pth_list.append(os.path.join(opt.logdir, 'net_latest.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        pca_path        = os.path.join(data_pca, os.path.splitext(os.path.basename(pth))[0])
        histogram_path  = os.path.join(data_histogram, os.path.splitext(os.path.basename(pth))[0])
        diff_path       = os.path.join(data_diff, os.path.splitext(os.path.basename(pth))[0])
        mat_path        = os.path.join(mat_base, os.path.splitext(os.path.basename(pth))[0])
        
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(pca_path, exist_ok=True)
        os.makedirs(histogram_path, exist_ok=True)
        os.makedirs(diff_path, exist_ok=True)
        os.makedirs(mat_path, exist_ok=True)

        print(pth)
        model.load_state_dict(torch.load(pth))
        model.eval()

        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []

        with torch.no_grad():
            for iteration, batch in enumerate(loader_valid, 1):
                feature_maps = []  # 重置特征图列表

                if opt.up_scale == 1:
                    gt, pan, lms = batch[0].cuda(), batch[3].cuda(), batch[1].cuda()

                else:
                    gt, pan, ms = batch[0].cuda(), batch[3].cuda(), batch[4].cuda()
                    # pixel shuffle
                    if opt.up_way == 'pixel_shuffle':
                        lms = F.pixel_shuffle(ms, upscale_factor=opt.up_scale)
                    # bicubic
                    elif opt.up_way == 'bicubic':
                        lms = F.interpolate(ms, scale_factor=opt.up_scale, mode='bicubic')
                    # bilinear
                    elif opt.up_way == 'bilinear':
                        lms = F.interpolate(ms, scale_factor=opt.up_scale, mode='bilinear')

                # gt, pan, lms = batch[0].cuda(), batch[3].cuda(), batch[1].cuda()
                sr, _ = model(lms, pan)         # w/o grad
                # metric
                gt = gt.detach().cpu().numpy()
                sr = sr.detach().cpu().numpy()

                ergas   = calc_ergas(gt, sr)
                psnr    = calc_psnr(gt, sr)
                rmse    = calc_rmse(gt, sr)
                sam     = calc_sam(gt, sr)
                scc     = calc_scc(gt, sr)
                q2n     = calc_qindex(gt, sr)
                epoch_val_ergas.append(ergas)
                epoch_val_psnr.append(psnr)
                epoch_val_rmse.append(rmse)
                epoch_val_sam.append(sam)
                epoch_val_scc.append(scc)
                epoch_val_q2n.append(q2n)
                # # save mat
                # savemat('{}.mat'.format(iteration), sr)
                # save rgb
                gt = np.squeeze(gt, 0)
                sr = np.squeeze(sr, 0)
                # diff = diff_gt_sr(gt, sr)
                if iteration < 21:
                    pca_analys(sr, os.path.join(pca_path, str(iteration)))
                    histogram_analys(sr, os.path.join(histogram_path, str(iteration)))
                    img_difference(gt, sr, os.path.join(diff_path, str(iteration)), vmin=0, vmax=opt.vmax)

                    gt_img = save_img(gt, opt.msi_channel)
                    sr_img = save_img(sr, opt.msi_channel)
                    # diff_img = save_img(diff)
                    gt_img_save = Image.fromarray(gt_img.astype(np.uint8), 'RGB')
                    sr_img_save = Image.fromarray(sr_img.astype(np.uint8), 'RGB')
                    # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
                    gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
                    sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))
                    mat_name = os.path.join(mat_path, "output_mulExm_" + str(iteration) + ".mat")
                    sio.savemat(mat_name, {'sr': sr})
                    # diff_img_save.save('{}/diff_{}.png'.format(rgb_path, iteration))

        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_ergas = np.nanmean(np.array(epoch_val_ergas))
        v_psnr  = np.nanmean(np.array(epoch_val_psnr))
        v_rmse  = np.nanmean(np.array(epoch_val_rmse))
        v_sam   = np.nanmean(np.array(epoch_val_sam))
        v_scc   = np.nanmean(np.array(epoch_val_scc))
        v_q2n   = np.nanmean(np.array(epoch_val_q2n))
        print('[pth:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}'.format(pth, v_ergas, v_psnr, v_rmse, v_sam, v_scc, v_q2n))

        with open('{}/{}.txt'.format(opt.save_path, opt.value_txt), 'a') as f:
            f.write(f'[pth:{pth}], SAM={v_sam}, RMSE={v_rmse}, ERGAS={v_ergas}, PSNR={v_psnr}, SCC={v_scc}, Q2n={v_q2n}\n')
        t_end   = time.time()
        print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        t_start = time.time()
    
    # writer.close()
    # 结束 wandb 运行
    wandb.finish()

def reduce_main():

    rgb_path_base   = os.path.join(opt.save_path, opt.img_save_path)
    data_pca        = os.path.join(opt.save_path, 'PCA')
    data_histogram  = os.path.join(opt.save_path, 'histogram')
    data_diff       = os.path.join(opt.save_path, 'diff')
    mat_base        = os.path.join(opt.save_path, 'mat')

    # os.makedirs(rgb_path_base, exist_ok=True)
    # os.makedirs(data_pca, exist_ok=True)
    # os.makedirs(data_histogram, exist_ok=True)


    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # Build model
    print('Loading model ...\n')
    model = opt.model_name(model_opt=opt)
    print_network(model)
    # grad_net = Get_gradient_nopadding_msi(opt.msi_channel)
    if opt.use_GPU:
        model = model.cuda()
    
    # t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
    pth_list.append(os.path.join(opt.logdir, 'net_latest.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        pca_path        = os.path.join(data_pca, os.path.splitext(os.path.basename(pth))[0])
        histogram_path  = os.path.join(data_histogram, os.path.splitext(os.path.basename(pth))[0])
        diff_path       = os.path.join(data_diff, os.path.splitext(os.path.basename(pth))[0])
        mat_path        = os.path.join(mat_base, os.path.splitext(os.path.basename(pth))[0])
        
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(pca_path, exist_ok=True)
        os.makedirs(histogram_path, exist_ok=True)
        os.makedirs(diff_path, exist_ok=True)
        os.makedirs(mat_path, exist_ok=True)

        print(pth)

        # model.load_state_dict(torch.load(pth))
        model = torch.load(pth)
        model.eval()

        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []

        total_time = 0.0

        with torch.no_grad():
            for iteration, batch in enumerate(loader_valid, 1):
                feature_maps = []  # 重置特征图列表

                if opt.up_scale == 1:
                    gt, pan, lms = batch[0].cuda(), batch[3].cuda(), batch[1].cuda()

                else:
                    gt, pan, ms = batch[0].cuda(), batch[3].cuda(), batch[4].cuda()
                    # pixel shuffle
                    if opt.up_way == 'pixel_shuffle':
                        lms = F.pixel_shuffle(ms, upscale_factor=opt.up_scale)
                    # bicubic
                    elif opt.up_way == 'bicubic':
                        lms = F.interpolate(ms, scale_factor=opt.up_scale, mode='bicubic')
                    # bilinear
                    elif opt.up_way == 'bilinear':
                        lms = F.interpolate(ms, scale_factor=opt.up_scale, mode='bilinear')

                # gt, pan, lms = batch[0].cuda(), batch[3].cuda(), batch[1].cuda()            
                torch.cuda.synchronize()
                start_time = time.time()

                sr, _, _ = model(lms, pan)    # grad  
                # sr, _ = model(lms, pan)         # w/o grad
                torch.cuda.synchronize()
                end_time = time.time()

                total_time += (end_time - start_time)

                # metric
                gt = gt.detach().cpu().numpy()
                sr = sr.detach().cpu().numpy()

                ergas   = calc_ergas(gt, sr)
                psnr    = calc_psnr(gt, sr)
                rmse    = calc_rmse(gt, sr)
                sam     = calc_sam(gt, sr)
                scc     = calc_scc(gt, sr)
                q2n     = calc_qindex(gt, sr)
                epoch_val_ergas.append(ergas)
                epoch_val_psnr.append(psnr)
                epoch_val_rmse.append(rmse)
                epoch_val_sam.append(sam)
                epoch_val_scc.append(scc)
                epoch_val_q2n.append(q2n)
                # # save mat
                # savemat('{}.mat'.format(iteration), sr)
                # save rgb
                gt = np.squeeze(gt, 0)
                sr = np.squeeze(sr, 0)
                # diff = diff_gt_sr(gt, sr)
                if iteration < 21:
                    pca_analys(sr, os.path.join(pca_path, str(iteration)))
                    histogram_analys(sr, os.path.join(histogram_path, str(iteration)))
                    img_difference(gt, sr, os.path.join(diff_path, str(iteration)), vmin=0, vmax=opt.vmax)

                    gt_img = save_img(gt, opt.msi_channel)
                    sr_img = save_img(sr, opt.msi_channel)
                    # diff_img = save_img(diff)
                    gt_img_save = Image.fromarray(gt_img.astype(np.uint8), 'RGB')
                    sr_img_save = Image.fromarray(sr_img.astype(np.uint8), 'RGB')
                    # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
                    gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
                    sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))
                    mat_name = os.path.join(mat_path, "output_mulExm_" + str(iteration) + ".mat")
                    sio.savemat(mat_name, {'sr': sr})
                    # diff_img_save.save('{}/diff_{}.png'.format(rgb_path, iteration))

        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_ergas = np.nanmean(np.array(epoch_val_ergas))
        v_psnr  = np.nanmean(np.array(epoch_val_psnr))
        v_rmse  = np.nanmean(np.array(epoch_val_rmse))
        v_sam   = np.nanmean(np.array(epoch_val_sam))
        v_scc   = np.nanmean(np.array(epoch_val_scc))
        v_q2n   = np.nanmean(np.array(epoch_val_q2n))
        print('[pth:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}|scc:{}|q2n:{}|time:{}'.format(pth, v_ergas, v_psnr, v_rmse, v_sam, v_scc, v_q2n, total_time))

        with open('{}/{}.txt'.format(opt.save_path, opt.value_txt), 'a') as f:
            f.write(f'[pth:{pth}], SAM={v_sam:.4f}, RMSE={v_rmse:.4f}, ERGAS={v_ergas:.4f}, PSNR={v_psnr:.4f}, SCC={v_scc:.4f}, Q2n={v_q2n:.4f}, time={total_time:.4f}\n')
        # t_end   = time.time()
        # print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        # t_start = time.time()
    
    # writer.close()
    # 结束 wandb 运行
    wandb.finish()

def WFANet_reduce():

    rgb_path_base   = os.path.join(opt.save_path, opt.img_save_path)
    data_pca        = os.path.join(opt.save_path, 'PCA')
    data_histogram  = os.path.join(opt.save_path, 'histogram')
    data_diff       = os.path.join(opt.save_path, 'diff')
    mat_base        = os.path.join(opt.save_path, 'mat')

    # os.makedirs(rgb_path_base, exist_ok=True)
    # os.makedirs(data_pca, exist_ok=True)
    # os.makedirs(data_histogram, exist_ok=True)


    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # Build model
    print('Loading model ...\n')
    model = opt.model_name(L_up_channel=opt.msi_channel, pan_channel=opt.pan_channel, ms_target_channel=32,
              pan_target_channel=32, head_channel=4, dropout= 0.085)
    print_network(model)
    # grad_net = Get_gradient_nopadding_msi(opt.msi_channel)
    if opt.use_GPU:
        model = model.cuda()
    
    # t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
    pth_list.append(os.path.join(opt.logdir, 'GF2_WFANet_epoch_latest.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        pca_path        = os.path.join(data_pca, os.path.splitext(os.path.basename(pth))[0])
        histogram_path  = os.path.join(data_histogram, os.path.splitext(os.path.basename(pth))[0])
        diff_path       = os.path.join(data_diff, os.path.splitext(os.path.basename(pth))[0])
        mat_path        = os.path.join(mat_base, os.path.splitext(os.path.basename(pth))[0])
        
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(pca_path, exist_ok=True)
        os.makedirs(histogram_path, exist_ok=True)
        os.makedirs(diff_path, exist_ok=True)
        os.makedirs(mat_path, exist_ok=True)

        print(pth)

        checkpoint = torch.load(pth)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []

        total_time = 0.0

        with torch.no_grad():
            for iteration, batch in enumerate(loader_valid, 1):
                feature_maps = []  # 重置特征图列表

                gt, pan, lms, ms = batch[0].cuda(), batch[3].cuda(), batch[1].cuda(), batch[4].cuda()

                # gt, pan, lms = batch[0].cuda(), batch[3].cuda(), batch[1].cuda()            
                torch.cuda.synchronize()
                start_time = time.time()

                sr = model(pan=pan, ms=ms, lms=lms)
                torch.cuda.synchronize()
                end_time = time.time()

                total_time += (end_time - start_time)

                # metric
                gt = gt.detach().cpu().numpy()
                sr = sr.detach().cpu().numpy()

                ergas   = calc_ergas(gt, sr)
                psnr    = calc_psnr(gt, sr)
                rmse    = calc_rmse(gt, sr)
                sam     = calc_sam(gt, sr)
                scc     = calc_scc(gt, sr)
                q2n     = calc_qindex(gt, sr)
                epoch_val_ergas.append(ergas)
                epoch_val_psnr.append(psnr)
                epoch_val_rmse.append(rmse)
                epoch_val_sam.append(sam)
                epoch_val_scc.append(scc)
                epoch_val_q2n.append(q2n)
                # # save mat
                # savemat('{}.mat'.format(iteration), sr)
                # save rgb
                gt = np.squeeze(gt, 0)
                sr = np.squeeze(sr, 0)
                # diff = diff_gt_sr(gt, sr)
                if iteration < 21:
                    pca_analys(sr, os.path.join(pca_path, str(iteration)))
                    histogram_analys(sr, os.path.join(histogram_path, str(iteration)))
                    img_difference(gt, sr, os.path.join(diff_path, str(iteration)), vmin=0, vmax=opt.vmax)

                    gt_img = save_img(gt, opt.msi_channel)
                    sr_img = save_img(sr, opt.msi_channel)
                    # diff_img = save_img(diff)
                    gt_img_save = Image.fromarray(gt_img.astype(np.uint8), 'RGB')
                    sr_img_save = Image.fromarray(sr_img.astype(np.uint8), 'RGB')
                    # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
                    gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
                    sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))
                    mat_name = os.path.join(mat_path, "output_mulExm_" + str(iteration) + ".mat")
                    sio.savemat(mat_name, {'sr': sr})
                    # diff_img_save.save('{}/diff_{}.png'.format(rgb_path, iteration))

        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_ergas = np.nanmean(np.array(epoch_val_ergas))
        v_psnr  = np.nanmean(np.array(epoch_val_psnr))
        v_rmse  = np.nanmean(np.array(epoch_val_rmse))
        v_sam   = np.nanmean(np.array(epoch_val_sam))
        v_scc   = np.nanmean(np.array(epoch_val_scc))
        v_q2n   = np.nanmean(np.array(epoch_val_q2n))
        print('[pth:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}|scc:{}|q2n:{}|time:{}'.format(pth, v_ergas, v_psnr, v_rmse, v_sam, v_scc, v_q2n, total_time))

        with open('{}/{}.txt'.format(opt.save_path, opt.value_txt), 'a') as f:
            f.write(f'[pth:{pth}], SAM={v_sam:.4f}, RMSE={v_rmse:.4f}, ERGAS={v_ergas:.4f}, PSNR={v_psnr:.4f}, SCC={v_scc:.4f}, Q2n={v_q2n:.4f}, time={total_time:.4f}\n')
        # t_end   = time.time()
        # print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        # t_start = time.time()
    
    # writer.close()
    # 结束 wandb 运行
    wandb.finish()

def BiMPan_reduce():

    rgb_path_base   = os.path.join(opt.save_path, opt.img_save_path)
    data_pca        = os.path.join(opt.save_path, 'PCA')
    data_histogram  = os.path.join(opt.save_path, 'histogram')
    data_diff       = os.path.join(opt.save_path, 'diff')
    mat_base        = os.path.join(opt.save_path, 'mat') 

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # Build model
    print('Loading model ...\n')
    model = opt.model_name(ms_channel=opt.msi_channel)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()

    # criterion = SSIM().cuda()
    
    t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
    pth_list.append(os.path.join(opt.logdir, 'net_latest.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        pca_path        = os.path.join(data_pca, os.path.splitext(os.path.basename(pth))[0])
        histogram_path  = os.path.join(data_histogram, os.path.splitext(os.path.basename(pth))[0])
        diff_path       = os.path.join(data_diff, os.path.splitext(os.path.basename(pth))[0])
        mat_path        = os.path.join(mat_base, os.path.splitext(os.path.basename(pth))[0])

        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(pca_path, exist_ok=True)
        os.makedirs(histogram_path, exist_ok=True)
        os.makedirs(diff_path, exist_ok=True)
        os.makedirs(mat_path, exist_ok=True)

        print(pth)
        model.load_state_dict(torch.load(pth))
        model.eval()
        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []

        total_time = 0.0

        for iteration, batch in enumerate(loader_valid, 1):
            if opt.up_scale == 1:
                gt, pan, lms = batch[0].cuda(), batch[3].cuda(), batch[1].cuda()

            else:
                gt, pan, ms = batch[0].cuda(), batch[3].cuda(), batch[4].cuda()
                # pixel shuffle
                if opt.up_way == 'pixel_shuffle':
                    lms = F.pixel_shuffle(ms, upscale_factor=opt.up_scale)
                # bicubic
                elif opt.up_way == 'bicubic':
                    lms = F.interpolate(ms, scale_factor=opt.up_scale, mode='bicubic')
                # bilinear
                elif opt.up_way == 'bilinear':
                    lms = F.interpolate(ms, scale_factor=opt.up_scale, mode='bilinear')
            
            # gt, pan, lms = batch[0].cuda(), batch[3].cuda(), batch[1].cuda()
            # sr, _, _ = model(lms, pan)    # grad
            torch.cuda.synchronize()
            start_time = time.time()

            sr = model(lms, pan)

            torch.cuda.synchronize()
            end_time = time.time()

            total_time += (end_time - start_time)
            # metric
            gt = gt.detach().cpu().numpy()
            sr = sr.detach().cpu().numpy()

            ergas   = calc_ergas(gt, sr)
            psnr    = calc_psnr(gt, sr)
            rmse    = calc_rmse(gt, sr)
            sam     = calc_sam(gt, sr)
            scc     = calc_scc(gt, sr)
            q2n     = calc_qindex(gt, sr)
            epoch_val_ergas.append(ergas)
            epoch_val_psnr.append(psnr)
            epoch_val_rmse.append(rmse)
            epoch_val_sam.append(sam)
            epoch_val_scc.append(scc)
            epoch_val_q2n.append(q2n)
            # # save mat
            # savemat('{}.mat'.format(iteration), sr)
            # save rgb
            gt = np.squeeze(gt, 0)
            sr = np.squeeze(sr, 0)
            # diff = diff_gt_sr(gt, sr)
            if iteration < 21:
                pca_analys(sr, os.path.join(pca_path, str(iteration)))
                histogram_analys(sr, os.path.join(histogram_path, str(iteration)))
                img_difference(gt, sr, os.path.join(diff_path, str(iteration)), vmin=0, vmax=opt.vmax)

                gt_img = save_img(gt, opt.msi_channel)
                sr_img = save_img(sr, opt.msi_channel)
                # diff_img = save_img(diff)
                gt_img_save = Image.fromarray(gt_img.astype(np.uint8))
                sr_img_save = Image.fromarray(sr_img.astype(np.uint8))
                # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
                gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
                sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))
                mat_name = os.path.join(mat_path, "output_mulExm_" + str(iteration) + ".mat")
                sio.savemat(mat_name, {'sr': sr})
                # diff_img_save.save('{}/diff_{}.png'.format(rgb_path, iteration))

        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_ergas = np.nanmean(np.array(epoch_val_ergas))
        v_psnr  = np.nanmean(np.array(epoch_val_psnr))
        v_rmse  = np.nanmean(np.array(epoch_val_rmse))
        v_sam   = np.nanmean(np.array(epoch_val_sam))
        v_scc   = np.nanmean(np.array(epoch_val_scc))
        v_q2n   = np.nanmean(np.array(epoch_val_q2n))
        print('[pth:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}'.format(pth, v_ergas, v_psnr, v_rmse, v_sam, v_scc, v_q2n))

        with open('{}/{}.txt'.format(opt.save_path, opt.value_txt), 'a') as f:
            f.write(f'[pth:{pth}], SAM={v_sam:.4f}, RMSE={v_rmse:.4f}, ERGAS={v_ergas:.4f}, PSNR={v_psnr:.4f}, SCC={v_scc:.4f}, Q2n={v_q2n:.4f}, time={total_time:.4f}\n')
        t_end   = time.time()
        print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        t_start = time.time()

def DCINN_reduce():
    rgb_path_base   = os.path.join(opt.save_path, opt.img_save_path)
    data_pca        = os.path.join(opt.save_path, 'PCA')
    data_histogram  = os.path.join(opt.save_path, 'histogram')
    data_diff       = os.path.join(opt.save_path, 'diff')
    mat_base        = os.path.join(opt.save_path, 'mat')

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # Build model
    print('Loading model ...\n')

    model = DCINN(channel_in=4, channel_out=4, block_num=3).cuda()

    t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
    pth_list.append(os.path.join(opt.logdir, 'net_latest.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        pca_path        = os.path.join(data_pca, os.path.splitext(os.path.basename(pth))[0])
        histogram_path  = os.path.join(data_histogram, os.path.splitext(os.path.basename(pth))[0])
        diff_path       = os.path.join(data_diff, os.path.splitext(os.path.basename(pth))[0])
        mat_path        = os.path.join(mat_base, os.path.splitext(os.path.basename(pth))[0])
        
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(pca_path, exist_ok=True)
        os.makedirs(histogram_path, exist_ok=True)
        os.makedirs(diff_path, exist_ok=True)
        os.makedirs(mat_path, exist_ok=True)

        checkpoint = torch.load(pth, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['f_model_state_dict'])

        model.eval()
        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []

        total_time = 0.0

        for iteration, batch in enumerate(loader_valid, 1):
            
            HSI, MS, HS = Variable(batch[0]), Variable(batch[1]), Variable(batch[3])
            HSI = HSI.cuda()
            MS  = MS.cuda()
            HS  = HS.cuda()

            MS = HSI
            MS0 = HSI
            HS0 = HS
            HS1 = torch.repeat_interleave(HS, 4, dim=1)

            torch.cuda.synchronize()
            start_time = time.time()

            out_HSI = model.forward(HS1-MS,MS0,HS0)+MS

            torch.cuda.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time)
 
            out_HSI = out_HSI.cpu().data.squeeze().clamp(0, 1).numpy()
            HSI = HSI.cpu().data.squeeze().clamp(0, 1).numpy()
            HS = HS.cpu().data.squeeze().clamp(0, 1).numpy()
            MS = MS.cpu().data.squeeze().clamp(0, 1).numpy()

            ergas   = calc_ergas(HSI, out_HSI)
            psnr    = calc_psnr(HSI, out_HSI)
            rmse    = calc_rmse(HSI, out_HSI)
            sam     = calc_sam(HSI, out_HSI)
            scc     = calc_scc(HSI, out_HSI)
            q2n     = calc_qindex(HSI, out_HSI)
            epoch_val_ergas.append(ergas)
            epoch_val_psnr.append(psnr)
            epoch_val_rmse.append(rmse)
            epoch_val_sam.append(sam)
            epoch_val_scc.append(scc)
            epoch_val_q2n.append(q2n)
            # # save mat
            # savemat('{}.mat'.format(iteration), out_HSI)
            # save rgb

            # HSI = np.squeeze(HSI, 0)
            # out_HSI = np.squeeze(out_HSI, 0)
            # diff = diff_gt_sr(HSI, out_HSI)

            gt_img = save_img(HSI, opt.msi_channel)
            sr_img = save_img(out_HSI, opt.msi_channel)

            # diff_img = save_img(diff)
            gt_img_save = Image.fromarray(gt_img.astype(np.uint8))
            sr_img_save = Image.fromarray(sr_img.astype(np.uint8))
            # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
            gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
            sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))
            mat_name = os.path.join(mat_path, "output_mulExm_" + str(iteration) + ".mat")
            sio.savemat(mat_name, {'sr': sr})
            # diff_img_save.save('{}/diff_{}.png'.format(rgb_path, iteration))

        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_ergas = np.nanmean(np.array(epoch_val_ergas))
        v_psnr  = np.nanmean(np.array(epoch_val_psnr))
        v_rmse  = np.nanmean(np.array(epoch_val_rmse))
        v_sam   = np.nanmean(np.array(epoch_val_sam))
        v_scc   = np.nanmean(np.array(epoch_val_scc))
        v_q2n   = np.nanmean(np.array(epoch_val_q2n))
        print('[pth:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}'.format(pth, v_ergas, v_psnr, v_rmse, v_sam, v_scc, v_q2n))

        with open('{}/{}.txt'.format(opt.save_path, opt.value_txt), 'a') as f:
            f.write(f'[pth:{pth}], SAM={v_sam:.4f}, RMSE={v_rmse:.4f}, ERGAS={v_ergas:.4f}, PSNR={v_psnr:.4f}, SCC={v_scc:.4f}, Q2n={v_q2n:.4f}, time={total_time:.4f}\n')
        t_end   = time.time()
        print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        t_start = time.time()

def TDNet_reduce():
    rgb_path_base   = os.path.join(opt.save_path, opt.img_save_path)
    data_pca        = os.path.join(opt.save_path, 'PCA')
    data_histogram  = os.path.join(opt.save_path, 'histogram')
    data_diff       = os.path.join(opt.save_path, 'diff') 
    mat_base        = os.path.join(opt.save_path, 'mat')

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # Build model
    print('Loading model ...\n')
    model = tdnet(channel=opt.msi_channel)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    
    t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
    pth_list.append(os.path.join(opt.logdir, '500.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        pca_path        = os.path.join(data_pca, os.path.splitext(os.path.basename(pth))[0])
        histogram_path  = os.path.join(data_histogram, os.path.splitext(os.path.basename(pth))[0])
        diff_path       = os.path.join(data_diff, os.path.splitext(os.path.basename(pth))[0])
        mat_path        = os.path.join(mat_base, os.path.splitext(os.path.basename(pth))[0])
        
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(pca_path, exist_ok=True)
        os.makedirs(histogram_path, exist_ok=True)
        os.makedirs(diff_path, exist_ok=True)
        os.makedirs(mat_path, exist_ok=True)
        print(pth)
        model.load_state_dict(torch.load(pth))
        model.eval()
        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []

        total_time = 0.0

        for iteration, batch in enumerate(loader_valid, 1):
            gt, lms, ms, pan = Variable(batch[0], requires_grad=False).cuda(), \
                                Variable(batch[1]).cuda(), \
                                Variable(batch[4]).cuda(), \
                                Variable(batch[3]).cuda()
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            ###GPU加速
            #pan = pan[:, np.newaxis,:, :].permute # expand to N*H*W*1
            out1, out2 = model(ms.float(), pan.float())  # call mode
            # print("out2 shape:", out2)
            # input("^.......")
            # print("out1 shape:", out1.shape)
            # print("gt shape:", gt)
            # input("^.......")
            sr = out2

            torch.cuda.synchronize()
            end_time = time.time()
            total_time += (end_time-start_time)
            
            # metric
            gt = gt.detach().cpu().numpy()
            sr = sr.detach().cpu().numpy()

            ergas   = calc_ergas(gt, sr)
            psnr    = calc_psnr(gt, sr)
            rmse    = calc_rmse(gt, sr)
            sam     = calc_sam(gt, sr)
            scc     = calc_scc(gt, sr)
            q2n     = calc_qindex(gt, sr)
            epoch_val_ergas.append(ergas)
            epoch_val_psnr.append(psnr)
            epoch_val_rmse.append(rmse)
            epoch_val_sam.append(sam)
            epoch_val_scc.append(scc)
            epoch_val_q2n.append(q2n)
            # # save mat
            # savemat('{}.mat'.format(iteration), sr)
            # save rgb
            gt = np.squeeze(gt, 0)
            sr = np.squeeze(sr, 0)
            if iteration < 21:
                # diff = diff_gt_sr(gt, sr)
                pca_analys(sr, os.path.join(pca_path, str(iteration)))
                histogram_analys(sr, os.path.join(histogram_path, str(iteration)))
                img_difference(gt, sr, os.path.join(diff_path, str(iteration)), vmin=0, vmax=opt.vmax)

                gt_img = save_img(gt, opt.msi_channel)
                sr_img = save_img(sr, opt.msi_channel)
                # diff_img = save_img(diff)
                gt_img_save = Image.fromarray(gt_img.astype(np.uint8))
                sr_img_save = Image.fromarray(sr_img.astype(np.uint8))
                # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
                gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
                sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))
                mat_name = os.path.join(mat_path, "output_mulExm_" + str(iteration) + ".mat")
                sio.savemat(mat_name, {'sr': sr})
                # diff_img_save.save('{}/diff_{}.png'.format(rgb_path, iteration))

        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_ergas = np.nanmean(np.array(epoch_val_ergas))
        v_psnr  = np.nanmean(np.array(epoch_val_psnr))
        v_rmse  = np.nanmean(np.array(epoch_val_rmse))
        v_sam   = np.nanmean(np.array(epoch_val_sam))
        v_scc   = np.nanmean(np.array(epoch_val_scc))
        v_q2n   = np.nanmean(np.array(epoch_val_q2n))
        print('[pth:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}'.format(pth, v_ergas, v_psnr, v_rmse, v_sam, v_scc, v_q2n))

        with open('{}/{}.txt'.format(opt.save_path, opt.value_txt), 'a') as f:
            f.write(f'[pth:{pth}], SAM={v_sam:.4f}, RMSE={v_rmse:.4f}, ERGAS={v_ergas:.4f}, PSNR={v_psnr:.4f}, SCC={v_scc:.4f}, Q2n={v_q2n:.4f}, time={total_time:.4f}\n')
        t_end   = time.time()
        print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        t_start = time.time()

def BDT_reduce():
    rgb_path_base = os.path.join(opt.save_path, opt.img_save_path)
    os.makedirs(rgb_path_base, exist_ok=True)

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # Build model
    print('Loading model ...\n')
    model = Bidinet(opt).cuda()
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    
    t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
    pth_list.append(os.path.join(opt.logdir, 'model_epoch_500.pth.tar'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        os.makedirs(rgb_path, exist_ok=True)
        print(pth)
        checkpoint = torch.load(pth)
        model.load_state_dict(checkpoint["model"].state_dict())     # 由于模型保存原因，这里将BDT的模型放在了model下，而不是sota_models下
        model.eval()
        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []
        for iteration, batch in enumerate(loader_valid, 1):
            pan, ms, input_lr_u, gt = Variable(batch[3]).cuda(), Variable(batch[4]).cuda(), Variable( batch[1]).cuda(), Variable(batch[0]).cuda()

            sr = model(pan, input_lr_u, ms)
            
            # metric
            gt = gt.detach().cpu().numpy()
            sr = sr.detach().cpu().numpy()

            ergas   = calc_ergas(gt, sr)
            psnr    = calc_psnr(gt, sr)
            rmse    = calc_rmse(gt, sr)
            sam     = calc_sam(gt, sr)
            scc     = calc_scc(gt, sr)
            q2n     = calc_qindex(gt, sr)
            epoch_val_ergas.append(ergas)
            epoch_val_psnr.append(psnr)
            epoch_val_rmse.append(rmse)
            epoch_val_sam.append(sam)
            epoch_val_scc.append(scc)
            epoch_val_q2n.append(q2n)
            # # save mat
            # savemat('{}.mat'.format(iteration), sr)
            # save rgb
            gt = np.squeeze(gt, 0)
            sr = np.squeeze(sr, 0)
            # diff = diff_gt_sr(gt, sr)

            gt_img = save_img(gt, opt.msi_channel)
            sr_img = save_img(sr, opt.msi_channel)
            # diff_img = save_img(diff)
            gt_img_save = Image.fromarray(gt_img.astype(np.uint8))
            sr_img_save = Image.fromarray(sr_img.astype(np.uint8))
            # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
            gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
            sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))
            # diff_img_save.save('{}/diff_{}.png'.format(rgb_path, iteration))

        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_ergas = np.nanmean(np.array(epoch_val_ergas))
        v_psnr  = np.nanmean(np.array(epoch_val_psnr))
        v_rmse  = np.nanmean(np.array(epoch_val_rmse))
        v_sam   = np.nanmean(np.array(epoch_val_sam))
        v_scc   = np.nanmean(np.array(epoch_val_scc))
        v_q2n   = np.nanmean(np.array(epoch_val_q2n))
        print('[pth:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}'.format(pth, v_ergas, v_psnr, v_rmse, v_sam, v_scc, v_q2n))

        with open('{}/{}.txt'.format(opt.save_path, opt.value_txt), 'a') as f:
            f.write(f'[pth:{pth}], SAM={v_sam:.4f}, RMSE={v_rmse:.4f}, ERGAS={v_ergas:.4f}, PSNR={v_psnr:.4f}, SCC={v_scc:.4f}, Q2n={v_q2n:.4f}, time={total_time:.4f}\n')
        t_end   = time.time()
        print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        t_start = time.time()

def U2Net_reduce():

    rgb_path_base = os.path.join(opt.save_path, opt.img_save_path)
    os.makedirs(rgb_path_base, exist_ok=True)

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # Build model
    print('Loading model ...\n')
    model = opt.model_name(dim = 32)
    print_network(model)
    # grad_net = Get_gradient_nopadding_msi(opt.msi_channel)
    if opt.use_GPU:
        model = model.cuda()

    # criterion = SSIM().cuda()
    
    t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'best*'))
    pth_list.append(os.path.join(opt.logdir, 'net_latest.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        os.makedirs(rgb_path, exist_ok=True)
        print(pth)
        model.load_state_dict(torch.load(pth))
        model.eval()
        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []
        for iteration, batch in enumerate(loader_valid, 1):
            gt, pan, ms = batch[0].cuda(), batch[3].cuda(), batch[4].cuda()

            sr = model(ms, pan)
            # metric
            gt = gt.detach().cpu().numpy()
            sr = sr.detach().cpu().numpy()

            ergas   = calc_ergas(gt, sr)
            psnr    = calc_psnr(gt, sr)
            rmse    = calc_rmse(gt, sr)
            sam     = calc_sam(gt, sr)
            scc     = calc_scc(gt, sr)
            q2n     = calc_qindex(gt, sr)
            epoch_val_ergas.append(ergas)
            epoch_val_psnr.append(psnr)
            epoch_val_rmse.append(rmse)
            epoch_val_sam.append(sam)
            epoch_val_scc.append(scc)
            epoch_val_q2n.append(q2n)
            # # save mat
            # savemat('{}.mat'.format(iteration), sr)
            # save rgb
            gt = np.squeeze(gt, 0)
            sr = np.squeeze(sr, 0)
            # diff = diff_gt_sr(gt, sr)

            gt_img = save_img(gt, opt.msi_channel)
            sr_img = save_img(sr, opt.msi_channel)
            # diff_img = save_img(diff)
            gt_img_save = Image.fromarray(gt_img.astype(np.uint8))
            sr_img_save = Image.fromarray(sr_img.astype(np.uint8))
            # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
            gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
            sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))
            # diff_img_save.save('{}/diff_{}.png'.format(rgb_path, iteration))

        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_ergas = np.nanmean(np.array(epoch_val_ergas))
        v_psnr  = np.nanmean(np.array(epoch_val_psnr))
        v_rmse  = np.nanmean(np.array(epoch_val_rmse))
        v_sam   = np.nanmean(np.array(epoch_val_sam))
        v_scc   = np.nanmean(np.array(epoch_val_scc))
        v_q2n   = np.nanmean(np.array(epoch_val_q2n))
        print('[pth:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}'.format(pth, v_ergas, v_psnr, v_rmse, v_sam, v_scc, v_q2n))

        with open('{}/{}.txt'.format(opt.save_path, opt.value_txt), 'a') as f:
            f.write(f'[pth:{pth}], SAM={v_sam:.4f}, RMSE={v_rmse:.4f}, ERGAS={v_ergas:.4f}, PSNR={v_psnr:.4f}, SCC={v_scc:.4f}, Q2n={v_q2n:.4f}, time={total_time:.4f}\n')
        t_end   = time.time()
        print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        t_start = time.time()

def Conv_reduce(): 

    rgb_path_base   = os.path.join(opt.save_path, opt.img_save_path)
    data_pca        = os.path.join(opt.save_path, 'PCA')
    data_histogram  = os.path.join(opt.save_path, 'histogram')
    data_diff       = os.path.join(opt.save_path, 'diff')
    mat_base        = os.path.join(opt.save_path, 'mat')

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # Build model
    print('Loading model ...\n')
    model = opt.model_name(channle=opt.msi_channel)
    print_network(model)
    # grad_net = Get_gradient_nopadding_msi(opt.msi_channel)
    if opt.use_GPU:
        model = model.cuda()

    # criterion = SSIM().cuda()
    
    t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
    pth_list.append(os.path.join(opt.logdir, '500.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        pca_path        = os.path.join(data_pca, os.path.splitext(os.path.basename(pth))[0])
        histogram_path  = os.path.join(data_histogram, os.path.splitext(os.path.basename(pth))[0])
        diff_path       = os.path.join(data_diff, os.path.splitext(os.path.basename(pth))[0])
        mat_path        = os.path.join(mat_base, os.path.splitext(os.path.basename(pth))[0])
        
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(pca_path, exist_ok=True)
        os.makedirs(histogram_path, exist_ok=True)
        os.makedirs(diff_path, exist_ok=True)
        os.makedirs(mat_path, exist_ok=True)

        print(pth)
        model.load_state_dict(torch.load(pth))
        model.eval()
        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []
        epoch_val_ssim  = []

        total_time = 0.0
        for iteration, batch in enumerate(loader_valid, 1):
            gt, lms, pan, ms = batch[0].cuda(), batch[1].cuda(), batch[3].cuda(), batch[4].cuda()         

            torch.cuda.synchronize()
            start_time = time.time()

            sr = model(lms, pan)

            torch.cuda.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time)
            # metric
            gt = gt.detach().cpu().numpy()
            sr = sr.detach().cpu().numpy()

            ergas   = calc_ergas(gt, sr)
            psnr    = calc_psnr(gt, sr)
            rmse    = calc_rmse(gt, sr)
            sam     = calc_sam(gt, sr)
            scc     = calc_scc(gt, sr)
            q2n     = calc_qindex(gt, sr)
            ssim    = calc_ssim(gt, sr)
            epoch_val_ergas.append(ergas)
            epoch_val_psnr.append(psnr)
            epoch_val_rmse.append(rmse)
            epoch_val_sam.append(sam)
            epoch_val_scc.append(scc)
            epoch_val_q2n.append(q2n)
            epoch_val_ssim.append(ssim)
            # # save mat
            # savemat('{}.mat'.format(iteration), sr)
            # save rgb
            gt = np.squeeze(gt, 0)
            sr = np.squeeze(sr, 0)

            if iteration < 21:
                # diff = diff_gt_sr(gt, sr)

                pca_analys(sr, os.path.join(pca_path, str(iteration)))
                histogram_analys(sr, os.path.join(histogram_path, str(iteration)))
                img_difference(gt, sr, os.path.join(diff_path, str(iteration)), vmin=0, vmax=opt.vmax)

                gt_img = save_img(gt, opt.msi_channel)
                sr_img = save_img(sr, opt.msi_channel)
                # diff_img = save_img(diff)
                gt_img_save = Image.fromarray(gt_img.astype(np.uint8))
                sr_img_save = Image.fromarray(sr_img.astype(np.uint8))
                # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
                gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
                sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))
                mat_name = os.path.join(mat_path, "output_mulExm_" + str(iteration) + ".mat")
                sio.savemat(mat_name, {'sr': sr})
                # diff_img_save.save('{}/diff_{}.png'.format(rgb_path, iteration))

        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_ergas = np.nanmean(np.array(epoch_val_ergas))
        v_psnr  = np.nanmean(np.array(epoch_val_psnr))
        v_rmse  = np.nanmean(np.array(epoch_val_rmse))
        v_sam   = np.nanmean(np.array(epoch_val_sam))
        v_scc   = np.nanmean(np.array(epoch_val_scc))
        v_q2n   = np.nanmean(np.array(epoch_val_q2n))
        v_ssim  = np.nanmean(np.array(epoch_val_ssim))
        print('[pth:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}|ssim:{}'.format(pth, v_ergas, v_psnr, v_rmse, v_sam, v_scc, v_q2n, v_ssim))

        with open('{}/{}.txt'.format(opt.save_path, opt.value_txt), 'a') as f:
            f.write(f'[pth:{pth}], PSNR={v_psnr:.4f}, RMSE={v_rmse:.4f}, SAM={v_sam:.4f}, ERGAS={v_ergas:.4f}, SCC={v_scc:.4f}, Q2n={v_q2n:.4f},SSIM={v_ssim},time={total_time:.4f} \n')
        t_end   = time.time()
        print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        t_start = time.time()

def CML_reduce():
    rgb_path_base   = os.path.join(opt.save_path, opt.img_save_path)
    data_pca        = os.path.join(opt.save_path, 'PCA')
    data_histogram  = os.path.join(opt.save_path, 'histogram')
    data_diff       = os.path.join(opt.save_path, 'diff')
    mat_base        = os.path.join(opt.save_path, 'mat')

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # Build model
    print('Loading model ...\n')
    model = opt.model_name(msi_channel=opt.msi_channel)
    print_network(model)
    # grad_net = Get_gradient_nopadding_msi(opt.msi_channel)
    if opt.use_GPU:
        model = model.cuda()

    # criterion = SSIM().cuda()
    
    t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
    pth_list.append(os.path.join(opt.logdir, '500.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        pca_path        = os.path.join(data_pca, os.path.splitext(os.path.basename(pth))[0])
        histogram_path  = os.path.join(data_histogram, os.path.splitext(os.path.basename(pth))[0])
        diff_path       = os.path.join(data_diff, os.path.splitext(os.path.basename(pth))[0])
        mat_path        = os.path.join(mat_base, os.path.splitext(os.path.basename(pth))[0])
        
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(pca_path, exist_ok=True)
        os.makedirs(histogram_path, exist_ok=True)
        os.makedirs(diff_path, exist_ok=True)
        os.makedirs(mat_path, exist_ok=True)
        print(pth)
        model.load_state_dict(torch.load(pth))
        model.eval()
        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []

        total_time = 0.0
        for iteration, batch in enumerate(loader_valid, 1):
            gt, lms, pan, ms = batch[0].cuda(), batch[1].cuda(), batch[3].cuda(), batch[4].cuda()

            torch.cuda.synchronize()
            start_time = time.time()

            sr = model(ms, pan)

            torch.cuda.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time)
            # metric
            gt = gt.detach().cpu().numpy()
            sr = sr.detach().cpu().numpy()

            ergas   = calc_ergas(gt, sr)
            psnr    = calc_psnr(gt, sr)
            rmse    = calc_rmse(gt, sr)
            sam     = calc_sam(gt, sr)
            scc     = calc_scc(gt, sr)
            q2n     = calc_qindex(gt, sr)
            epoch_val_ergas.append(ergas)
            epoch_val_psnr.append(psnr)
            epoch_val_rmse.append(rmse)
            epoch_val_sam.append(sam)
            epoch_val_scc.append(scc)
            epoch_val_q2n.append(q2n)
            # # save mat
            # savemat('{}.mat'.format(iteration), sr)
            # save rgb
            gt = np.squeeze(gt, 0)
            sr = np.squeeze(sr, 0)
            if iteration < 21:
                # diff = diff_gt_sr(gt, sr)
                pca_analys(sr, os.path.join(pca_path, str(iteration)))
                histogram_analys(sr, os.path.join(histogram_path, str(iteration)))
                img_difference(gt, sr, os.path.join(diff_path, str(iteration)), vmin=0, vmax=opt.vmax)
    
                gt_img = save_img(gt, opt.msi_channel)
                sr_img = save_img(sr, opt.msi_channel)
                # diff_img = save_img(diff)
                gt_img_save = Image.fromarray(gt_img.astype(np.uint8))
                sr_img_save = Image.fromarray(sr_img.astype(np.uint8))
                # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
                gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
                sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))
                mat_name = os.path.join(mat_path, "output_mulExm_" + str(iteration) + ".mat")
                sio.savemat(mat_name, {'sr': sr})
                # diff_img_save.save('{}/diff_{}.png'.format(rgb_path, iteration))

        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_ergas = np.nanmean(np.array(epoch_val_ergas))
        v_psnr  = np.nanmean(np.array(epoch_val_psnr))
        v_rmse  = np.nanmean(np.array(epoch_val_rmse))
        v_sam   = np.nanmean(np.array(epoch_val_sam))
        v_scc   = np.nanmean(np.array(epoch_val_scc))
        v_q2n   = np.nanmean(np.array(epoch_val_q2n))
        print('[pth:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}'.format(pth, v_ergas, v_psnr, v_rmse, v_sam, v_scc, v_q2n))

        with open('{}/{}.txt'.format(opt.save_path, opt.value_txt), 'a') as f:
            f.write(f'[pth:{pth}], SAM={v_sam:.4f}, RMSE={v_rmse:.4f}, ERGAS={v_ergas:.4f}, PSNR={v_psnr:.4f}, SCC={v_scc:.4f}, Q2n={v_q2n:.4f}, time={total_time:.4f}\n')
        t_end   = time.time()
        print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        t_start = time.time()

def GPPNN_reduce():

    rgb_path_base   = os.path.join(opt.save_path, opt.img_save_path)
    data_pca        = os.path.join(opt.save_path, 'PCA')
    data_histogram  = os.path.join(opt.save_path, 'histogram')
    data_diff       = os.path.join(opt.save_path, 'diff')
    mat_base        = os.path.join(opt.save_path, 'mat')

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # Build model
    print('Loading model ...\n')
    model = opt.model_name(ms_channels=opt.msi_channel,
           pan_channels=opt.pan_channel, 
           n_feat=opt.kernel_channel,
           n_layer=8)
    print_network(model)
    # grad_net = Get_gradient_nopadding_msi(opt.msi_channel)
    if opt.use_GPU:
        model = model.cuda()

    # criterion = SSIM().cuda()
    
    t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
    pth_list.append(os.path.join(opt.logdir, 'last_net.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        pca_path        = os.path.join(data_pca, os.path.splitext(os.path.basename(pth))[0])
        histogram_path  = os.path.join(data_histogram, os.path.splitext(os.path.basename(pth))[0])
        diff_path       = os.path.join(data_diff, os.path.splitext(os.path.basename(pth))[0])
        mat_path        = os.path.join(mat_base, os.path.splitext(os.path.basename(pth))[0])
        
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(pca_path, exist_ok=True)
        os.makedirs(histogram_path, exist_ok=True)
        os.makedirs(diff_path, exist_ok=True)
        os.makedirs(mat_path, exist_ok=True)

        print(pth)
        checkpoint = torch.load(pth)
        model.load_state_dict(checkpoint['net'])

        model.eval()
        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []
        for iteration, batch in enumerate(loader_valid, 1):
            gt, lms, pan, ms = batch[0].cuda(), batch[1].cuda(), batch[3].cuda(), batch[4].cuda()

            sr = model(ms, pan)
            # metric
            gt = gt.detach().cpu().numpy()
            sr = sr.detach().cpu().numpy()

            ergas   = calc_ergas(gt, sr)
            psnr    = calc_psnr(gt, sr)
            rmse    = calc_rmse(gt, sr)
            sam     = calc_sam(gt, sr)
            scc     = calc_scc(gt, sr)
            q2n     = calc_qindex(gt, sr)
            epoch_val_ergas.append(ergas)
            epoch_val_psnr.append(psnr)
            epoch_val_rmse.append(rmse)
            epoch_val_sam.append(sam)
            epoch_val_scc.append(scc)
            epoch_val_q2n.append(q2n)
            # # save mat
            # savemat('{}.mat'.format(iteration), sr)
            # save rgb
            gt = np.squeeze(gt, 0)
            sr = np.squeeze(sr, 0)
            # diff = diff_gt_sr(gt, sr)

            if iteration < 21:
                pca_analys(sr, os.path.join(pca_path, str(iteration)))
                histogram_analys(sr, os.path.join(histogram_path, str(iteration)))
                img_difference(gt, sr, os.path.join(diff_path, str(iteration)), vmin=0, vmax=opt.vmax)

                gt_img = save_img(gt, opt.msi_channel)
                sr_img = save_img(sr, opt.msi_channel)
                # diff_img = save_img(diff)
                gt_img_save = Image.fromarray(gt_img.astype(np.uint8))
                sr_img_save = Image.fromarray(sr_img.astype(np.uint8))
                # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
                gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
                sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))
                mat_name = os.path.join(mat_path, "output_mulExm_" + str(iteration) + ".mat")
                sio.savemat(mat_name, {'sr': sr})
                # diff_img_save.save('{}/diff_{}.png'.format(rgb_path, iteration))

        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_ergas = np.nanmean(np.array(epoch_val_ergas))
        v_psnr  = np.nanmean(np.array(epoch_val_psnr))
        v_rmse  = np.nanmean(np.array(epoch_val_rmse))
        v_sam   = np.nanmean(np.array(epoch_val_sam))
        v_scc   = np.nanmean(np.array(epoch_val_scc))
        v_q2n   = np.nanmean(np.array(epoch_val_q2n))
        print('[pth:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}'.format(pth, v_ergas, v_psnr, v_rmse, v_sam, v_scc, v_q2n))

        with open('{}/{}.txt'.format(opt.save_path, opt.value_txt), 'a') as f:
            f.write(f'[pth:{pth}], SAM={v_sam:.4f}, RMSE={v_rmse:.4f}, ERGAS={v_ergas:.4f}, PSNR={v_psnr:.4f}, SCC={v_scc:.4f}, Q2n={v_q2n:.4f}, time={total_time:.4f}\n')
        t_end   = time.time()
        print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        t_start = time.time()

def UTeRM_reduce():

    rgb_path_base   = os.path.join(opt.save_path, opt.img_save_path)
    data_pca        = os.path.join(opt.save_path, 'PCA')
    data_histogram  = os.path.join(opt.save_path, 'histogram')
    data_diff       = os.path.join(opt.save_path, 'diff')
    mat_base        = os.path.join(opt.save_path, 'mat')

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=False)

    # Build model
    print('Loading model ...\n')
    model = opt.model_name(HSI_channels=opt.msi_channel)
    print_network(model)
    # grad_net = Get_gradient_nopadding_msi(opt.msi_channel)
    if opt.use_GPU:
        model = model.cuda()

    # criterion = SSIM().cuda()
    
    t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
    pth_list.append(os.path.join(opt.logdir, 'net_latest.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        pca_path        = os.path.join(data_pca, os.path.splitext(os.path.basename(pth))[0])
        histogram_path  = os.path.join(data_histogram, os.path.splitext(os.path.basename(pth))[0])
        diff_path       = os.path.join(data_diff, os.path.splitext(os.path.basename(pth))[0])
        mat_path        = os.path.join(mat_base, os.path.splitext(os.path.basename(pth))[0])
        
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(pca_path, exist_ok=True)
        os.makedirs(histogram_path, exist_ok=True)
        os.makedirs(diff_path, exist_ok=True)
        os.makedirs(mat_path, exist_ok=True)

        print(pth)
        checkpoint = torch.load(pth)
        model.load_state_dict(checkpoint)

        model.eval()
        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []
        for iteration, batch in enumerate(loader_valid, 1):
            gt, lms, pan, ms = batch[0].cuda(), batch[1].cuda(), batch[3].cuda(), batch[4].cuda()

            _, sr = model(lms, pan)
            # metric
            gt = gt.detach().cpu().numpy()
            sr = sr.detach().cpu().numpy()

            ergas   = calc_ergas(gt, sr)
            psnr    = calc_psnr(gt, sr)
            rmse    = calc_rmse(gt, sr)
            sam     = calc_sam(gt, sr)
            scc     = calc_scc(gt, sr)
            q2n     = calc_qindex(gt, sr)
            epoch_val_ergas.append(ergas)
            epoch_val_psnr.append(psnr)
            epoch_val_rmse.append(rmse)
            epoch_val_sam.append(sam)
            epoch_val_scc.append(scc)
            epoch_val_q2n.append(q2n)
            # # save mat
            # savemat('{}.mat'.format(iteration), sr)
            # save rgb
            gt = np.squeeze(gt, 0)
            sr = np.squeeze(sr, 0)
            # diff = diff_gt_sr(gt, sr)

            if iteration < 21:
                pca_analys(sr, os.path.join(pca_path, str(iteration)))
                histogram_analys(sr, os.path.join(histogram_path, str(iteration)))
                img_difference(gt, sr, os.path.join(diff_path, str(iteration)), vmin=0, vmax=opt.vmax)

                gt_img = save_img(gt, opt.msi_channel)
                sr_img = save_img(sr, opt.msi_channel)
                # diff_img = save_img(diff)
                gt_img_save = Image.fromarray(gt_img.astype(np.uint8))
                sr_img_save = Image.fromarray(sr_img.astype(np.uint8))
                # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
                gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
                sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))
                mat_name = os.path.join(mat_path, "output_mulExm_" + str(iteration) + ".mat")
                sio.savemat(mat_name, {'sr': sr})
                # diff_img_save.save('{}/diff_{}.png'.format(rgb_path, iteration))

        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_ergas = np.nanmean(np.array(epoch_val_ergas))
        v_psnr  = np.nanmean(np.array(epoch_val_psnr))
        v_rmse  = np.nanmean(np.array(epoch_val_rmse))
        v_sam   = np.nanmean(np.array(epoch_val_sam))
        v_scc   = np.nanmean(np.array(epoch_val_scc))
        v_q2n   = np.nanmean(np.array(epoch_val_q2n))
        print('[pth:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}'.format(pth, v_ergas, v_psnr, v_rmse, v_sam, v_scc, v_q2n))

        with open('{}/{}.txt'.format(opt.save_path, opt.value_txt), 'a') as f:
            f.write(f'[pth:{pth}], SAM={v_sam:.4f}, RMSE={v_rmse:.4f}, ERGAS={v_ergas:.4f}, PSNR={v_psnr:.4f}, SCC={v_scc:.4f}, Q2n={v_q2n:.4f}, time={total_time:.4f}\n')
        t_end   = time.time()
        print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        t_start = time.time()

if __name__ == "__main__":
    if opt.data_type == 'PReNetGradient' or opt.data_type == 'PReNetGradientFuse' or opt.data_type == 'PReNetGradientFuse_':
        reduce_main()
        # visual_grad()
    elif opt.data_type == 'PReNetSCAMOne':
        ablation_main()
    elif opt.data_type == 'DCINN':
        DCINN_reduce()
    elif opt.data_type == 'TDNet':
        TDNet_reduce()
    elif opt.data_type == 'BDT':
        BDT_reduce()
    elif opt.data_type == 'U2Net':
        U2Net_reduce()
    elif opt.data_type == 'LGPConv' or opt.data_type == 'CANNet':
        Conv_reduce()
    elif opt.data_type == 'LAGConv':
        Conv_reduce()
    elif opt.data_type == 'CML':
        CML_reduce()
    elif opt.data_type == 'GPPNN':
        GPPNN_reduce()
    elif opt.data_type == 'BiMPan':
        BiMPan_reduce()
    elif opt.data_type == 'UTeRM':
        UTeRM_reduce()
    elif opt.data_type == 'WFANet':
        WFANet_reduce()