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

import wandb
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from thop import profile

def model_type(model_name):
    ########## Proposed Method ##########
    if model_name == "GGPNet":
        return GGPNet
    else:
        raise ValueError(f"Unknown model name: {model_name}")

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="/data3/lianglanyue/PReNet_PAN1/logs/0702/lightgcmv2_gf2", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="../Datasets/GF2_Deng/test_h5/test_multiExm1.h5", help='path to training data')
parser.add_argument("--save_path", type=str, default="results/NCrebuttal/0715/gf2_light_test", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument("--model_name", type=model_type, default='GGPNet', help='choose model')
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--msi_channel", type=int, default=4, help='MSI channel')
parser.add_argument("--pan_channel", type=int, default=1, help='PAN channel')
parser.add_argument("--kernel_channel", type=int, default=64, help='convolutional output behind the image input')   # 在服务器上计算时修改为64
parser.add_argument("--data_type", type=str, default='GGPNet', help='same as model_name')
parser.add_argument("--img_save_path", type=str, default='RGB_img')
parser.add_argument("--value_txt", type=str, default='value_info')
parser.add_argument("--vmax", type=float , default=0.1, help='diff max value')
# Upsample
parser.add_argument("--up_scale", type=int, default=1, help='if up_scale=1, MSI input id=lms; else MSI input id=ms')
parser.add_argument("--up_way", type=str, default='bicubic', help='pixel shuffle; bicubic; bilinear')

opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# 相关性计算
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

if __name__ == "__main__":
    if opt.data_type == 'GGPNet':
        reduce_main()
        # visual_grad()