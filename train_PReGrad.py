import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from model.networks import *
from model.block import *
from load_train_data import Dataset_Pro
from metrics import *
import glob
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import time
import datetime


import argparse

def model_type(model_name):
    if model_name == "PReNet_MSI":
        return PReNet_MSI
    elif model_name == "PReNet_PAN":
        return PReNet_PAN
    # elif model_name == "PRN":
    #     return PRN
    elif model_name == "PReNetSCAMOne":
        return PReNetSCAMOne
    elif model_name == "PReNetSCAMResTwo":
        return PReNetSCAMResTwo
    elif model_name == "PReNetSCAMTwo":
        return PReNetSCAMTwo
    elif model_name == "PReNetS2Block":
        return PReNetS2Block
    elif model_name == "PReNetGradient":
        return PReNetGradient
    elif model_name == "PReNetGradientFuse":
        return PReNetGradientFuse
    elif model_name == "PReNetGradientFuse_":
        return PReNetGradientFuse_

    elif model_name == "ResNetGradient":
        return ResNetGradient
    elif model_name == "PReNetDiffGradient":
        return PReNetDiffGradient
    else:
        raise ValueError(f"Unknown model name: {model_name}")

parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[100,200,300], help="When to decay learning rate")    # [30, 50, 80]
parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--gpu_id", type=str, default='0', help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages') # 6
parser.add_argument("--eval_freq", type=int, default=100, help='frequency of evaluation and save model')
parser.add_argument("--optimizer", type=str, default='Adam', help='Adam, AdamW')
parser.add_argument('--seed', type=int, default='3047', help='seed number')
# Path
parser.add_argument("--save_path", type=str, default='logs/0709/gcm_gf2_grad_log', help='path to save models and log files')
parser.add_argument("--data_path",type=str, default='../Datasets/GF2_Deng/train.h5',help='path to training data')
parser.add_argument("--valid_path",type=str, default='../Datasets/GF2_Deng/test_h5/test_multiExm1.h5',help='path to valid data')
# Channel info
parser.add_argument("--msi_channel", type=int, default=4, help='MSI channel')
parser.add_argument("--pan_channel", type=int, default=1, help='PAN channel')
parser.add_argument("--kernel_channel", type=int, default=32, help='convolutional output behind the image input')   # 在服务器上计算时修改为64 light改为32
parser.add_argument("--model_name", type=model_type, default='PReNetDiffGradient', help='choose model')
# Loss info
parser.add_argument("--grad_loss", type=float, default=0.2, help='proportion of the grad_loss')
parser.add_argument("--pixel_loss", type=float, default=0.8, help='proportion of the metric_loss')
# Upsample
parser.add_argument("--up_scale", type=int, default=1, help='if up_scale=1, MSI input id=lms; else MSI input id=ms')
parser.add_argument("--up_way", type=str, default='bilinear', help='pixel shuffle; bicubic; bilinear')
opt = parser.parse_args()
    
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# device = 'cuda:'+ opt.gpu_id

# torch.manual_seed(opt.seed)
# torch.cuda.manual_seed(opt.seed)
# torch.cuda.manual_seed_all(opt.seed)
# cudnn.deterministic = True

def main():

    print('Loading dataset ...\n')
    dataset_train = Dataset_Pro(file_path=opt.data_path)
    dataset_valid = Dataset_Pro(file_path=opt.valid_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    print("# of testing samples: %d\n" % int(len(dataset_valid)))

    # Build model
    # model = PReNet(recurrent_iter=opt.recurrent_iter, gpu_id=device, msi_c=opt.msi_channel, pan_c=opt.pan_channel).cuda()
    model = opt.model_name(model_opt=opt).cuda()
    print("------ model name:{} ------\n".format(opt.model_name))
    print_network(model)
    # grad_net = Get_gradient_nopadding_msi(opt.msi_channel)
    grad_net = GradientExtractor()

    # loss function
    # criterion2 = nn.MSELoss(size_average=False)
    criterion2 = SSIM().cuda()
    criterion1 = nn.L1Loss()

    # Optimizer
    if opt.optimizer   == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == 'AdamW': 
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # record training
    writer = SummaryWriter(opt.save_path)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))
        model = torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch))

    # start training
    t_start = time.time()
    print('Start training...')
    step1 = 0
    step2 = 0
    best_sam    = float('inf')
    best_ergas  = float('inf')
    best_psnr   = -float('inf')
    with open('{}/value_info.txt'.format(opt.save_path), 'a') as f:
                f.write(f'arg:\n{opt}\n')

    for epoch in range(initial_epoch, opt.epochs):
        model.train()
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        epoch_train_loss = []
        ## epoch training start
        for i, input_train in enumerate(loader_train, 0):

            # time
            iter_start = time.time()

            model.zero_grad()
            optimizer.zero_grad()

            if opt.up_scale == 1:
                gt, pan, lms = input_train[0].cuda(), input_train[3].cuda(), input_train[1].cuda()

            else:
                gt, pan, ms = input_train[0].cuda(), input_train[3].cuda(), input_train[4].cuda()
                # pixel shuffle
                if opt.up_way == 'pixel_shuffle':
                    lms = F.pixel_shuffle(ms, upscale_factor=opt.up_scale)
                # bicubic
                elif opt.up_way == 'bicubic':
                    lms = F.interpolate(ms, scale_factor=opt.up_scale, mode='bicubic')
                # bilinear
                elif opt.up_way == 'bilinear':
                    lms = F.interpolate(ms, scale_factor=opt.up_scale, mode='bilinear')
              
            out_train, _, out_grad = model(lms, pan)
            gt_grad = grad_net(gt)

            # 都用ssim loss
            grad_metric = criterion2(gt_grad, out_grad)
            pixel_metric = criterion2(gt, out_train)
            loss = -(opt.pixel_loss*pixel_metric + opt.grad_loss*grad_metric)

            # # test 0220 grad用ssim loss，pixel用l1loss
            # grad_metric = -criterion2(gt_grad, out_grad)
            # pixel_metric = criterion1(gt, out_train)
            # loss = (opt.pixel_loss*pixel_metric + opt.grad_loss*grad_metric)

            # # test 0221 都用l1loss
            # grad_metric = criterion1(gt_grad, out_grad)
            # pixel_metric = criterion1(gt, out_train)
            # loss = (opt.pixel_loss*pixel_metric + opt.grad_loss*grad_metric)

            # test 0225 grad用l1loss，pixel用ssimloss
            # grad_metric = criterion1(gt_grad, out_grad)
            # pixel_metric = -criterion2(gt, out_train)
            # loss = (opt.pixel_loss*pixel_metric + opt.grad_loss*grad_metric)

            epoch_train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item() ))
            writer.add_scalar('itera_loss', loss.item(), step1)

            iter_end = time.time()
            elapsed = iter_end - iter_start
            total_iters = opt.epochs * len(loader_train)
            current_iter = epoch * len(loader_train) + i + 1
            remaining_iters = total_iters - current_iter
            time_per_iter = elapsed / current_iter
            eta_seconds = int(remaining_iters * time_per_iter)
            eta = str(datetime.timedelta(seconds=eta_seconds))
    
            print(f"[Epoch {epoch+1}/{opt.epochs}] [Iter {i+1}/{len(loader_train)}] "
                  f"Loss: {loss.item():.4f} | Time/iter: {time_per_iter:.2f}s | ETA: {eta}")
            # training curve
            # model.eval()
            # out_train, _ = model(input_train, pan)
            # out_train = torch.clamp(out_train, 0., 1.)
            # psnr_train = batch_PSNR(out_train, gt, 1.)
            # print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
            #       (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            # if step % 10 == 0:
            #     # Log the scalar values
            #     writer.add_scalar('loss', loss.item(), step)
            #     writer.add_scalar('PSNR on training data', psnr_train, step)
            step1 += 1
        ## epoch training end

        print("【epoch %d】 loss: %.4f, pixel_metric: %.4f" %
              (epoch+1, np.nanmean(np.array(epoch_train_loss)), pixel_metric.item()))
        writer.add_scalar('epoch_loss', np.nanmean(np.array(epoch_train_loss)), step2)
        step2 += 1

        # log the images
        if (epoch+1) % opt.eval_freq == 0:
            model.eval()
            epoch_val_loss  = []
            epoch_val_ergas = []
            epoch_val_psnr  = []
            epoch_val_rmse  = []
            epoch_val_sam   = []
            epoch_val_scc   = []
            epoch_val_q2n   = []
            with torch.no_grad():
                for iteration, batch in enumerate(loader_valid, 1):
                    gt, pan, lms = batch[0].cuda(), batch[3].cuda(), batch[1].cuda()
                    sr, _, out_grad  = model(lms, pan)
                    gt_grad = grad_net(gt)
                    grad_metric = criterion2(out_grad, gt_grad)
                    pixel_metric = criterion2(sr, gt)
                    loss = -(opt.pixel_loss*pixel_metric + opt.grad_loss*grad_metric)
                    epoch_val_loss.append(loss.item())

                    # metric
                    gt = gt.detach().cpu().numpy()
                    sr = sr.detach().cpu().numpy()
                    ergas   = calc_ergas(gt, sr)
                    psnr    = calc_psnr(gt, sr)
                    rmse    = calc_rmse(gt, sr)
                    sam     = calc_sam(gt, sr)
                    # scc     = calc_scc(gt, sr)
                    # q2n     = calc_qindex(gt, sr)      

                    epoch_val_ergas.append(ergas)
                    epoch_val_psnr.append(psnr)
                    epoch_val_rmse.append(rmse)
                    epoch_val_sam.append(sam)
                    # epoch_val_scc.append(scc)
                    # epoch_val_q2n.append(q2n)

                v_loss  = np.nanmean(np.array(epoch_val_loss))
                print('---------------validate loss: {:.7f}---------------'.format(v_loss))
                v_ergas = np.nanmean(np.array(epoch_val_ergas))
                v_psnr  = np.nanmean(np.array(epoch_val_psnr))
                v_rmse  = np.nanmean(np.array(epoch_val_rmse))
                v_sam   = np.nanmean(np.array(epoch_val_sam))
                # v_scc   = np.nanmean(np.array(epoch_val_scc))
                # v_q2n   = np.nanmean(np.array(epoch_val_q2n))

                print('[epoch:{}] ergas:{}|psnr:{}|rmse:{}|sam:{}'.format(epoch+1, v_ergas, v_psnr, v_rmse, v_sam))
                with open('{}/value_info.txt'.format(opt.save_path), 'a') as f:
                    f.write(f'epoch={epoch+1}, SAM={v_sam:.4f}, RMSE={v_rmse:.4f}, ERGAS={v_ergas:.4f}, PSNR={v_psnr:.4f}\n')
                t_end   = time.time()
                print('---------------total time cost: {:.4f}s--------------------'.format(t_end - t_start))
                t_start = time.time()


                # out_train, _ = model(input_train, pan)
                # out_train = torch.clamp(out_train, 0., 1.)
                # im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
                # im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
                # im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
                # writer.add_image('clean image', im_target, epoch+1)
                # writer.add_image('rainy image', im_input, epoch+1)
                # writer.add_image('deraining image', im_derain, epoch+1)

                torch.save(model, os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))
                # save best SAM epoch
                if v_sam <= best_sam:
                    if glob.glob(os.path.join(opt.save_path, 'best_sam_net_epoch*')) != []:
                        os.remove(glob.glob(os.path.join(opt.save_path, 'best_sam_net_epoch*'))[0])
                        # print("this epoch sam: {} <= last epoch best sam: {}, remove last best sam net".format(v_sam, best_sam))
                    torch.save(model.state_dict(), os.path.join(opt.save_path, 'best_sam_net_epoch%d.pth' % (epoch+1)))
                    best_sam = v_sam
                    print("save best sam net")
                # save best ERGAS epoch
                if v_ergas <= best_ergas:
                    if glob.glob(os.path.join(opt.save_path, 'best_ergas_net_epoch*')) != []:
                        os.remove(glob.glob(os.path.join(opt.save_path, 'best_ergas_net_epoch*'))[0])
                        # print("this epoch ergas: {} <= last epoch best ergas: {}, remove last best ergas net".format(v_ergas, best_ergas))
                    torch.save(model.state_dict(), os.path.join(opt.save_path, 'best_ergas_net_epoch%d.pth' % (epoch+1)))
                    best_ergas = v_ergas
                    print("save best ergas net")
                # save best PSNR epoch
                if v_psnr >= best_psnr:
                    if glob.glob(os.path.join(opt.save_path, 'best_psnr_net_epoch*')) != []:
                        os.remove(glob.glob(os.path.join(opt.save_path, 'best_psnr_net_epoch*'))[0])
                        # print("this epoch psnr: {} >= last epoch best ergas: {}, remove last best ergas net".format(v_ergas, best_ergas))
                    torch.save(model.state_dict(), os.path.join(opt.save_path, 'best_psnr_net_epoch%d.pth' % (epoch+1)))
                    best_psnr = v_psnr
                    print("save best psnr net")
        # save model
        # torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        torch.save(model, os.path.join(opt.save_path, 'net_latest.pth'))
        # if epoch % opt.save_freq == 0:
        #     torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":

    main()
