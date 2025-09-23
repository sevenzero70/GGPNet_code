import torch
import numpy as np
import cv2
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim


def calc_ssim(img_tgt, img_fus):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    
    Args:
        img_tgt (numpy.ndarray): First image (grayscale or color).
        img_fus (numpy.ndarray): Second image (same shape as img1).
    
    Returns:
        float: SSIM value.
    """
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    # # Ensure the input images are in the range [0, 255] for correct SSIM calculation
    # if img_tgt.max() > 1.0 or img_fus.max() > 1.0:
    #     img_tgt = img_tgt.astype(np.float32) / 255.0
    #     img_fus = img_fus.astype(np.float32) / 255.0
    min_size = min(img_tgt.shape[:2])  # Get the smallest dimension
    win_size = 3 if min_size < 7 else 7  # Adjust win_size based on image size

    # Compute SSIM
    ssim_value, _ = ssim(img_tgt, img_fus, full=True, multichannel=(img_tgt.ndim == 3), win_size=win_size, data_range=1.0)
    return ssim_value

def calc_ergas(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt-img_fus)**2, axis=1)
    rmse = rmse**0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse/mean)**2)
    ergas = 100/4*ergas**0.5

    return ergas

def calc_psnr(img_tgt, img_fus):

    # mse = np.mean((img_tgt-img_fus)**2)
    # img_max = np.max(img_tgt)
    # psnr = 10*np.log10(img_max**2/mse)
    # return psnr

    nc = img_tgt.shape[0]
    psnr =0
    for i in range(nc):
        mse = np.mean((img_tgt[i,:,:]- img_fus[i,:,:])** 2)
        psnr += 10 * np.log10(1/mse)

    return psnr/nc

def calc_rmse(img_tgt, img_fus):
    rmse = np.sqrt(np.mean((img_tgt-img_fus)**2))

    return rmse

def calc_sam(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt / np.max(img_tgt)
    img_fus = img_fus / np.max(img_fus)

    A = np.sqrt(np.sum(img_tgt**2, axis=0))
    B = np.sqrt(np.sum(img_fus**2, axis=0))
    AB = np.sum(img_tgt*img_fus, axis=0)

    sam = AB/(A*B+1e-7)     # 避免除零
    # 限制 sam 数组中的值在 [-1, 1] 范围内
    sam = np.clip(sam, -1, 1)
    sam = np.arccos(sam)
    sam = np.mean(sam)*180/3.1415926535

    return sam

# def sam(img1, img2):
#     """SAM for (N, C, H, W) image; torch.float32 [0.,1.]."""
#     inner_product = (img1 * img2).sum(dim=1)
#     img1_spectral_norm = torch.sqrt((img1**2).sum(dim=1))
#     img2_spectral_norm = torch.sqrt((img2**2).sum(dim=1))
#     # numerical stability
#     cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + eps)).clamp(min=0, max=1)
#     cos_theta = cos_theta.reshape(cos_theta.shape[0], -1)
#     return torch.mean(torch.acos(cos_theta), dim=-1).mean()

def calc_scc(img1, img2):

    """SCC for 2D (H, W)or 3D (H, W, C) image; uint or float[0, 1]"""
    if not  img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if img1.ndim == 4:
        img1 = np.transpose(np.squeeze(img1, axis=0), (1,2,0))
        img2 = np.transpose(np.squeeze(img2, axis=0), (1,2,0))
    if img1.ndim == 2:
        return np.corrcoef(img1.reshape(1, -1), img2.rehshape(1, -1))[0, 1]
    elif img1.ndim == 3:
        # print(img1[..., i].reshape[1, -1].shape)
        #test = np.corrcoef(img1_[..., i].reshape[1, -1], img2_[..., i].rehshape(1, -1))
        #print(type(test))
        scc = [np.corrcoef(img1[..., i].reshape(1, -1), img2[..., i].reshape(1, -1))[0, 1]
               for i in range(img1.shape[2])]
        # scc_list = []  # 初始化一个空列表来存储每个通道的空间相关系数（SCC）
        # # 遍历每个通道
        # for i in range(img1.shape[2]):
        #     # 为当前通道的两个图像计算相关系数矩阵
        #     img1_flat = img1[..., i].reshape(1, -1)
        #     img2_flat = img2[..., i].reshape(1, -1)
        #     std1 = np.std(img1_flat)
        #     std2 = np.std(img2_flat)
        #     print("std1", std1)
        #     print("std2", std2)
        #     input()
        #     if std1 == 0 or std2 == 0:
        #         print("1")
        #         corr_coefficient = np.nan
        #     else:
        #         # 如果标准差非零，安全地计算相关系数
        #         corr_matrix = np.corrcoef(img1_flat, img2_flat)
        #         corr_coefficient = corr_matrix[0, 1]
        #     scc_list.append(corr_coefficient)
        #     scc = np.array(scc_list)

        # print("scc", scc)
        # input("1111")
        return np.nanmean(scc)
    else:
        raise ValueError('Wrong input image dimensions.')

def _qindex(img1, img2, block_size=8):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    assert block_size > 1, 'block_size shold be greater than 1!'
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size**2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size/2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu2_sq
#    print(mu1_mu2.shape)
    #print(sigma2_sq.shape)
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0
    
#    print(np.min(sigma1_sq + sigma2_sq), np.min(mu1_sq + mu2_sq))
    
    idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) >1e-8)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0
    idx = ((sigma1_sq + sigma2_sq) >1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) >1e-8) * ((mu1_sq + mu2_sq) >1e-8)
    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    
    return np.mean(qindex_map)


def calc_qindex(img1, img2, block_size=8):
    """Q-index for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 4:
        img1 = np.transpose(np.squeeze(img1, axis=0), (1,2,0))
        img2 = np.transpose(np.squeeze(img2, axis=0), (1,2,0))
    if img1.ndim == 2:
        return _qindex(img1, img2, block_size)
    elif img1.ndim == 3:
        qindexs = [_qindex(img1[..., i], img2[..., i], block_size) for i in range(img1.shape[2])]
        return np.array(qindexs).mean()
    else:
        raise ValueError('Wrong input image dimensions.')
    

##################
# No reference IQA
##################

###### prepare ######

def gaussian2d(N, std):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std)**2) * np.exp(-0.5 * (t2 / std)**2) 
    return w


def kaiser2d(N, beta):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2) / np.double(N - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0
    return w


def fir_filter_wind(Hd, w):
    """
    compute fir (finite impulse response) filter with window method
    Hd: desired freqeuncy response (2D)
    w: window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = h / np.sum(h)
    return h

def GNyq2win(GNyq, scale=4, N=41):
    """Generate a 2D convolutional window from a given GNyq
    GNyq: Nyquist frequency
    scale: spatial size of PAN / spatial size of MS
    """
    #fir filter with window method
    fcut = 1 / scale
    alpha = np.sqrt(((N - 1) * (fcut / 2))**2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = fir_filter_wind(Hd, w)
    return np.real(h)

def mtf_resize(img, satellite='QuickBird', scale=4):
    # satellite GNyq
    scale = int(scale)
    if satellite == 'QuickBird':
        GNyq = [0.34, 0.32, 0.30, 0.22]  # Band Order: B,G,R,NIR 采样频率，找不到对应卫星对应通道的采样频率
        GNyqPan = 0.15
    elif satellite == 'IKONOS':
        GNyq = [0.26, 0.28, 0.29, 0.28]  # Band Order: B,G,R,NIR
        GNyqPan = 0.17
    elif satellite == 'A':
        GNyq = [0.3, 0.2, 0.5, 0.4]  # Band Order: B,G,R,NIR
        GNyqPan = 0.19
    else:
        raise NotImplementedError('satellite: QuickBird or IKONOS')
    # lowpass
    img_ = img.squeeze()
    img_ = img_.astype(np.float64)
    if img_.ndim == 2:  # Pan
        H, W = img_.shape
        lowpass = GNyq2win(GNyqPan, scale, N=41)
    elif img_.ndim == 3:  # MS
        H, W, _ = img.shape
        lowpass = [GNyq2win(gnyq, scale, N=41) for gnyq in GNyq]
        lowpass = np.stack(lowpass, axis=-1)
    img_ = ndimage.filters.correlate(img_, lowpass, mode='nearest')
    # downsampling
    output_size = (H // scale, W // scale)
    img_ = cv2.resize(img_, dsize=output_size, interpolation=cv2.INTER_NEAREST)
    return img_

###############################

def calc_D_lambda(img_fake, img_lm, block_size=32, p=1):
    """Spectral distortion
    img_fake, generated HRMS
    img_lm, LRMS"""
    assert img_fake.ndim == img_lm.ndim == 3, 'Images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # D_lambda
    Q_fake = []
    Q_lm = []
    for i in range(C_f):
        for j in range(i+1, C_f):
            # for fake
            band1 = img_fake[..., i]
            band2 = img_fake[..., j]
            Q_fake.append(_qindex(band1, band2, block_size=block_size))
            # for real
            band1 = img_lm[..., i]
            band2 = img_lm[..., j]
            Q_lm.append(_qindex(band1, band2, block_size=block_size))
    Q_fake = np.array(Q_fake)
    Q_lm = np.array(Q_lm)
    D_lambda_index = (np.abs(Q_fake - Q_lm) ** p).mean()
    return D_lambda_index ** (1/p)


def calc_D_s(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, q=1):
    """Spatial distortion
    img_fake, generated HRMS
    img_lm, LRMS
    pan, HRPan"""
    # fake and lm
    assert img_fake.ndim == img_lm.ndim == 3, 'MS images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert H_f // H_r == W_f // W_r == scale, 'Spatial resolution should be compatible with scale'
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # fake and pan
    assert pan.ndim == 3, 'Panchromatic image must be 3D!'
    H_p, W_p, C_p = pan.shape
    assert C_p == 1, 'size of 3rd dim of Panchromatic image must be 1'
    assert H_f == H_p and W_f == W_p, "Pan's and fake's spatial resolution should be the same"
    # get LRPan, 2D
    pan_lr = mtf_resize(pan, satellite=satellite, scale=scale)
    # pan_lr = np.squeeze(pan, -1)
    #print(pan_lr.shape)
    # D_s
    Q_hr = []
    Q_lr = []
    for i in range(C_f):
        # for HR fake
        band1 = img_fake[..., i]
        band2 = pan[..., 0] # the input PAN is 3D with size=1 along 3rd dim
        # print(band1.shape)
        # print(band2.shape)
        Q_hr.append(_qindex(band1, band2, block_size=block_size))
        band1 = img_lm[..., i]
        band2 = pan_lr  # this is 2D
        #print(band1.shape)
        #print(band2.shape)
        Q_lr.append(_qindex(band1, band2, block_size=block_size))
    Q_hr = np.array(Q_hr)
    Q_lr = np.array(Q_lr)
    D_s_index = (np.abs(Q_hr - Q_lr) ** q).mean()
    return D_s_index ** (1/q)

def calc_qnr(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, p=1, q=1, alpha=1, beta=1):
    """QNR - No reference IQA"""
    D_lambda_idx = calc_D_lambda(img_fake, img_lm, block_size, p)
    D_s_idx = calc_D_s(img_fake, img_lm, pan, satellite, scale, block_size, q)
    QNR_idx = (1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta
    return QNR_idx