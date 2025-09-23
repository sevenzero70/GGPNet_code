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

from sota_models.BiMPan.BiMPan import *
from sota_models.DCINN.dcinn_ps import *
from sota_models.TDNet.model_8band import tdnet
from model.model_SR_x4 import *
from sota_models.U2Net.u2net import *
from sota_models.LGPConv.model import *
from sota_models.LAGConv.model import *
from sota_models.CML.model import *
from sota_models.GPPNN.GPPNN import *

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
    ########## Proposed Method ##########
    elif model_name == "PReNetGradient":
        return PReNetGradient
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
    else:
        raise ValueError(f"Unknown model name: {model_name}")

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="/data3/lianglanyue/PReNet_PAN1/logs/0708/gcm_gf2", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="../Datasets/GF2_Deng/test_h5/test_OrigScale_multiExm1.h5", help='path to training data')
parser.add_argument("--save_path", type=str, default="results/NCrebuttal/0716/gcm_gf2_0708", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="2", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=1, help='number of recursive stages') # 6
parser.add_argument("--model_name", type=model_type, default='PReNetGradient', help='choose model')
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--msi_channel", type=int, default=4, help='MSI channel')
parser.add_argument("--pan_channel", type=int, default=1, help='PAN channel')
parser.add_argument("--kernel_channel", type=int, default=64, help='convolutional output behind the image input')   # 在服务器上计算时修改为64
parser.add_argument("--data_type", type=str, default='PReNetGradient', help='reduce, full')
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
            # lms = F.interpolate(lms, scale_factor=1/4, mode='bilinear')
            # pan = F.interpolate(pan, scale_factor=1/4, mode='bilinear')
            # ms  = F.interpolate(ms, scale_factor=1/4, mode='bilinear')
            # print("lms shape", lms.shape)
            # print("pan shape", pan.shape)
            # print("ms shape", ms.shape)
            # input("!!!!!")
            
            sr, _, _ = model(lms, pan)
            # sr = model(lms, pan)    # BiMPan
            # sr = model(ms, pan)     # CML
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

def BiMPan_full():

    rgb_path_base = os.path.join(opt.save_path, opt.img_save_path)
    os.makedirs(rgb_path_base, exist_ok=True)

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=True)

    # Build model
    print('Loading model ...\n')
    model = opt.model_name(model_opt=opt)
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
        os.makedirs(rgb_path, exist_ok=True)
        print(pth)
        model.load_state_dict(torch.load(pth))
        model.eval()
        epoch_val_D_lambda  = []
        epoch_val_D_s       = []
        epoch_val_qnr       = []
        for iteration, batch in enumerate(loader_valid, 1):
            if opt.up_scale == 1:
                ms, pan, lms = batch[4].cuda(), batch[3].cuda(), batch[1].cuda()

            else:
                ms, pan, lms = batch[4].cuda(), batch[3].cuda(), batch[1].cuda()
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
            sr = model(lms, pan)
            # metric
            sr  = sr.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            lms = lms.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            pan = pan.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            ms  = ms.squeeze(0).permute(1,2,0).detach().cpu().numpy()

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

def DCINN_full():
    rgb_path_base = os.path.join(opt.save_path, opt.img_save_path)
    os.makedirs(rgb_path_base, exist_ok=True)

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=True)

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
        os.makedirs(rgb_path, exist_ok=True)

        checkpoint = torch.load(pth, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['f_model_state_dict'])

        model.eval()
        epoch_val_ergas = []
        epoch_val_psnr  = []
        epoch_val_rmse  = []
        epoch_val_sam   = []
        epoch_val_scc   = []
        epoch_val_q2n   = []
        for iteration, batch in enumerate(loader_valid, 1):
            
            HSI, MS, HS = Variable(batch[0]), Variable(batch[1]), Variable(batch[3])
            HSI = HSI.cuda()
            MS  = MS.cuda()
            HS  = HS.cuda()

            MS = HSI
            MS0 = HSI
            HS0 = HS
            HS1 = torch.repeat_interleave(HS, 4, dim=1)
            out_HSI = model.forward(HS1-MS,MS0,HS0)+MS
 
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

def TDNet_full():
    rgb_path_base   = os.path.join(opt.save_path, opt.img_save_path)
    data_pca        = os.path.join(opt.save_path, 'PCA')
    data_histogram  = os.path.join(opt.save_path, 'histogram')
    data_diff       = os.path.join(opt.save_path, 'diff') 

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=True)

    # Build model
    print('Loading model ...\n')
    model = tdnet()
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    
    t_start = time.time()
    # Test each saved epoch
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
    pth_list.append(os.path.join(opt.logdir, 'net_latest.pth'))
    print(pth_list)
    for pth in pth_list:
        rgb_path = os.path.join(rgb_path_base, os.path.splitext(os.path.basename(pth))[0])
        pca_path        = os.path.join(data_pca, os.path.splitext(os.path.basename(pth))[0])
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(pca_path, exist_ok=True)
        print(pth)
        model.load_state_dict(torch.load(pth))
        model.eval()
        epoch_val_D_lambda  = []
        epoch_val_D_s       = []
        epoch_val_qnr       = []
        for iteration, batch in enumerate(loader_valid, 1):
            gt, lms, ms, pan = Variable(batch[0], requires_grad=False).cuda(), \
                                Variable(batch[1]).cuda(), \
                                Variable(batch[4]).cuda(), \
                                Variable(batch[3]).cuda()
            ###GPU加速
            #pan = pan[:, np.newaxis,:, :].permute # expand to N*H*W*1
            out1, out2 = model(ms.float(), pan.float())  # call mode

            sr = out2

            # metric
            sr  = sr.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            lms = lms.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            pan = pan.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            ms  = ms.squeeze(0).permute(1,2,0).detach().cpu().numpy()

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

def U2Net_full():

    rgb_path_base = os.path.join(opt.save_path, opt.img_save_path)
    os.makedirs(rgb_path_base, exist_ok=True)

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=True)

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
    pth_list = glob.glob(os.path.join(opt.logdir, 'nobest*'))
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
            f.write(f'[pth:{pth}], SAM={v_sam}, RMSE={v_rmse}, ERGAS={v_ergas}, PSNR={v_psnr}, SCC={v_scc}, Q2n={v_q2n}\n')
        t_end   = time.time()
        print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
        t_start = time.time()

def Conv_full(): 

    rgb_path_base = os.path.join(opt.save_path, opt.img_save_path)
    os.makedirs(rgb_path_base, exist_ok=True)

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=True)

    # Build model
    print('Loading model ...\n')
    model = opt.model_name()
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
        os.makedirs(rgb_path, exist_ok=True)
        print(pth)
        model.load_state_dict(torch.load(pth))
        model.eval()
        epoch_val_D_lambda  = []
        epoch_val_D_s       = []
        epoch_val_qnr       = []
        for iteration, batch in enumerate(loader_valid, 1):
            gt, lms, pan, ms = batch[0].cuda(), batch[1].cuda(), batch[3].cuda(), batch[4].cuda()

            sr = model(lms, pan)
            # metric
            sr  = sr.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            lms = lms.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            pan = pan.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            ms  = ms.squeeze(0).permute(1,2,0).detach().cpu().numpy()

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

def CML_full():

    rgb_path_base = os.path.join(opt.save_path, opt.img_save_path)
    os.makedirs(rgb_path_base, exist_ok=True)

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=True)

    # Build model
    print('Loading model ...\n')
    model = opt.model_name()
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
        os.makedirs(rgb_path, exist_ok=True)
        print(pth)
        model.load_state_dict(torch.load(pth))
        model.eval()
        epoch_val_D_lambda  = []
        epoch_val_D_s       = []
        epoch_val_qnr       = []
        for iteration, batch in enumerate(loader_valid, 1):
            lms, pan, ms = batch[1].cuda(), batch[3].cuda(), batch[4].cuda()

            sr = model(ms, pan)
            # metric
            sr  = sr.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            lms = lms.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            pan = pan.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            ms  = ms.squeeze(0).permute(1,2,0).detach().cpu().numpy()

            D_lambda    = calc_D_lambda(sr, ms)
            D_s         = calc_D_s(sr, ms, pan)
            qnr         = (1 - D_lambda) ** 1 * (1 - D_s) ** 1
            epoch_val_D_lambda.append(D_lambda)
            epoch_val_D_s.append(D_s)
            epoch_val_qnr.append(qnr)
            # # save mat
            # savemat('{}.mat'.format(iteration), sr)
            # save rgb
            # gt = np.squeeze(gt, 0)
            # sr = np.squeeze(sr, 0)
            # gt_img = save_img(gt, opt.msi_channel)
            # sr_img = save_img(sr, opt.msi_channel)
            # gt_img_save = Image.fromarray(gt_img.astype(np.uint8))
            # sr_img_save = Image.fromarray(sr_img.astype(np.uint8))
            # gt_img_save.save('{}/gt_{}.png'.format(rgb_path, iteration))
            # sr_img_save.save('{}/sr_{}.png'.format(rgb_path, iteration))

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

def GPPNN_full():

    rgb_path_base = os.path.join(opt.save_path, opt.img_save_path)
    os.makedirs(rgb_path_base, exist_ok=True)

    dataset_valid = Dataset_Pro(file_path=opt.data_path)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=True)

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
        os.makedirs(rgb_path, exist_ok=True)
        print(pth)
        checkpoint = torch.load(pth)
        model.load_state_dict(checkpoint['net'])

        model.eval()
        epoch_val_D_lambda  = []
        epoch_val_D_s       = []
        epoch_val_qnr       = []
        for iteration, batch in enumerate(loader_valid, 1):
            gt, lms, pan, ms = batch[0].cuda(), batch[1].cuda(), batch[3].cuda(), batch[4].cuda()

            sr = model(ms, pan)
            # metric
            sr  = sr.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            lms = lms.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            pan = pan.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            ms  = ms.squeeze(0).permute(1,2,0).detach().cpu().numpy()

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
    if opt.data_type == 'PReNetGradient':
        full_main()
    elif opt.data_type == 'DCINN':
        DCINN_full()
    elif opt.data_type == 'TDNet':
        TDNet_full()
    elif opt.data_type == 'BDT':
        BDT_full()
    elif opt.data_type == 'U2Net':
        U2Net_full()
    elif opt.data_type == 'LGPConv':
        Conv_full()
    elif opt.data_type == 'LAGConv':
        Conv_full()
    elif opt.data_type == 'CML':
        CML_full()
    elif opt.data_type == 'GPPNN':
        GPPNN_full()
    elif opt.data_type == 'BiMPan':
        BiMPan_full()