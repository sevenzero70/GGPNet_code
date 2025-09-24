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
from load_train_data import *
from torch.utils.data import DataLoader
import re
from SSIM import SSIM
from metrics import *
from PIL import Image

def model_type(model_name):
    ########## Proposed Method ##########
    if model_name == "GGPNet":
        return GGPNet
    else:
        raise ValueError(f"Unknown model name: {model_name}")

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="/data3/lianglanyue/PReNet_PAN1/logs/0708/gcm_gf2", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="../Datasets/GF2_Deng/test_h5/test_OrigScale_multiExm1.h5", help='path to training data')
parser.add_argument("--save_path", type=str, default="results/NCrebuttal/0716/gcm_gf2_0708", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="2", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=1, help='number of recursive stages') # 6
parser.add_argument("--model_name", type=model_type, default='GGPNet', help='choose model')
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--msi_channel", type=int, default=4, help='MSI channel')
parser.add_argument("--pan_channel", type=int, default=1, help='PAN channel')
parser.add_argument("--kernel_channel", type=int, default=64, help='convolutional output behind the image input')   # 在服务器上计算时修改为64
parser.add_argument("--data_type", type=str, default='GGPNet', help='reduce, full')
parser.add_argument("--img_save_path", type=str, default='GF2_RGB_img')
parser.add_argument("--value_txt", type=str, default='value_info')
# Upsample
parser.add_argument("--up_scale", type=int, default=1, help='if up_scale=1, MSI input id=lms; else MSI input id=ms')
parser.add_argument("--up_way", type=str, default='bicubic', help='pixel shuffle; bicubic; bilinear')

opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def full_main():


    rgb_path_base   = os.path.join(opt.save_path, opt.img_save_path)
    data_pca        = os.path.join(opt.save_path, 'PCA')
    data_histogram  = os.path.join(opt.save_path, 'histogram')
    data_diff       = os.path.join(opt.save_path, 'diff') 

    dataset_valid = Full_Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=1, batch_size=opt.batch_size, shuffle=True)


    # Build model
    print('Loading model ...\n')
    model = opt.model_name(model_opt=opt)     # BiMPan\Grad
    # model = opt.model_name()
    print_network(model)
    # grad_net = Get_gradient_nopadding_msi(opt.msi_channel)
    if opt.use_GPU:
        model = model.cuda()

    criterion = SSIM().cuda()
    
    t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
    pth_list.append(os.path.join(opt.logdir, 'net_latest.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        pca_path        = os.path.join(data_pca, os.path.splitext(os.path.basename(pth))[0])
        # histogram_path  = os.path.join(data_histogram, os.path.splitext(os.path.basename(pth))[0])
        # diff_path       = os.path.join(data_diff, os.path.splitext(os.path.basename(pth))[0])
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(pca_path, exist_ok=True)
        # os.makedirs(histogram_path, exist_ok=True)
        # os.makedirs(diff_path, exist_ok=True)
        
        print(pth)
        # model.load_state_dict(torch.load(pth))
        model = torch.load(pth)
        model.eval()
        epoch_val_D_lambda  = []
        epoch_val_D_s       = []
        epoch_val_qnr       = []
        for iteration, batch in enumerate(loader_valid, 1):
            if opt.up_scale == 1:
                ms, pan, lms = batch[2].cuda(), batch[1].cuda(), batch[0].cuda()

            else:
                ms, pan, lms = batch[2].cuda(), batch[1].cuda(), batch[0].cuda()
                # pixel shuffle
                if opt.up_way == 'pixel_shuffle':
                    lms = F.pixel_shuffle(ms, upscale_factor=opt.up_scale)
                # bicubic
                elif opt.up_way == 'bicubic':
                    lms = F.interpolate(ms, scale_factor=opt.up_scale, mode='bicubic')
                # bilinear
                elif opt.up_way == 'bilinear':
                    lms = F.interpolate(ms, scale_factor=opt.up_scale, mode='bilinear')

            sr, _, _ = model(lms, pan)

            # metric
            sr  = sr.squeeze(0).detach().cpu().numpy()
            lms = lms.squeeze(0).detach().cpu().numpy()
            pan = pan.squeeze(0).detach().cpu().numpy()
            ms  = ms.squeeze(0).detach().cpu().numpy()

            pca_analys(sr, os.path.join(pca_path, str(iteration)))
            # histogram_analys(sr, os.path.join(histogram_path, str(iteration)))
            # img_difference(gt, sr, os.path.join(diff_path, str(iteration)))

            # gt_img = save_img(gt, opt.msi_channel)
            sr_img = save_img(sr, opt.msi_channel)
            # diff_img = save_img(diff)

            # gt_img_save = Image.fromarray(gt_img.astype(np.uint8), 'RGB')
            sr_img_save = Image.fromarray(sr_img.astype(np.uint8), 'RGB')
            # diff_img_save = Image.fromarray(diff_img.astype(np.uint8))
            # gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
            sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))

            sr   = np.transpose(sr, (1, 2, 0))
            ms   = np.transpose(ms, (1, 2, 0))
            pan   = np.transpose(pan, (1, 2, 0))
            
            D_lambda    = calc_D_lambda(sr, ms)
            D_s         = calc_D_s(sr, ms, pan)
            qnr         = (1 - D_lambda) ** 1 * (1 - D_s) ** 1
            epoch_val_D_lambda.append(D_lambda)
            epoch_val_D_s.append(D_s)
            epoch_val_qnr.append(qnr)
        # v_loss  = np.nanmean(np.array(epoch_val_loss))
        # print('---------------validate loss: {:.7f}---------------'.format(v_loss))
        v_d_lambda  = np.nanmean(np.array(epoch_val_D_lambda))
        v_d_s       = np.nanmean(np.array(epoch_val_D_s))
        v_qnr       = np.nanmean(np.array(epoch_val_qnr))
        print('[pth:{}] D_lambda:{}|D_s:{}|QNR:{}'.format(pth, v_d_lambda, v_d_s, v_qnr))

        with open('{}/value_info.txt'.format(opt.save_path), 'a') as f:
            f.write(f'[pth:{pth}], D_lambda={v_d_lambda}, D_s={v_d_s}, QNR={v_qnr}\n')
        t_end   = time.time()
        print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        t_start = time.time()

if __name__ == "__main__":
    if opt.data_type == 'GGPNet':
        full_main()