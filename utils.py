import math
import torch
import re
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import  os
import glob

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from matplotlib.colors import Normalize


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])


def normalize(data):
    return data / 255.


def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return  False


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_img(np_img, m_channel):
    if m_channel == 8:
        np_img_r = np_img[4,:,:]  # red chaanel
        np_img_g = np_img[2,:,:]  # green channel
        np_img_b = np_img[1,:,:]  # blue channel
    else:
        np_img_r = np_img[2,:,:]  # red chaanel
        np_img_g = np_img[1,:,:]  # green channel
        np_img_b = np_img[0,:,:]  # blue channel
    out_img = np.stack([np_img_r, np_img_g, np_img_b], axis=-1)
    out_img = (out_img / out_img.max()) * 255
    out_img = np.clip(out_img, 0, 255)
    # out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min())
    # out_img = (out_img * 255).astype(np.uint8)
    return out_img

## data analysis
def histogram_analys(multi_spectral_image, save_name):

    # 绘制每个波段的直方图
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 8))
    axes = axes.ravel()

    for i in range(multi_spectral_image.shape[0]):
        ax = axes[i]
        ax.hist(multi_spectral_image[i, :, :].ravel(), bins=256, color='blue', alpha=0.7)
        ax.set_title(f'Band {i+1} Histogram')

    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=300)
    # plt.show()
    plt.close()

def pca_analys(multi_spectral_image, save_name):
    multi_spectral_image = np.transpose(multi_spectral_image,(1, 2, 0))
    data_reshaped = multi_spectral_image.reshape(multi_spectral_image.shape[0] * multi_spectral_image.shape[1],
                                                 multi_spectral_image.shape[2])
    pca = PCA(n_components=1)
    data_pca = pca.fit_transform(data_reshaped) 
    data_pca_orginshape = data_pca.reshape(multi_spectral_image.shape[0], multi_spectral_image.shape[1], 1)
    # for i in range(3):
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(data_pca_orginshape[:, :, i], cmap='inferno')
    #     plt.colorbar()
    #     plt.title(f'{save_name}_{i+1}.png')
    #     plt.axis('off')
    #     plt.savefig(f'{save_name}_{i+1}.png', bbox_inches='tight')
    #     plt.close()  # 关闭图形，防止它们在 Jupyter 笔记本中显示
    plt.figure(figsize=(8, 8))
    plt.imshow(data_pca_orginshape[:, :, 0], cmap='inferno')
    plt.colorbar()
    # plt.title(f'{save_name}.png')
    plt.axis('off')
    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    plt.close()  # 关闭图形，防止它们在 Jupyter 笔记本中显示

def img_difference(image1, image2, save_name, vmin=0, vmax=0.1):

    # 计算两个图像之间的差异
    image_difference = abs(image1 - image2)

    average_difference = np.mean(image_difference, axis=0)

    # 创建一个归一化对象，统一颜色条范围
    norm = Normalize(vmin=vmin, vmax=vmax)

    # 显示平均后的单通道差异图像
    plt.imshow(average_difference, cmap='cool', norm=norm, interpolation='none')
    plt.colorbar()  # 显示颜色条
    # plt.title('Average Difference Across All Channels')
    # plt.show()
    plt.axis('off')
    plt.savefig(f'{save_name}.png', bbox_inches='tight')
    plt.close()  # 关闭图形，防止它们在 Jupyter 笔记本中显示
 