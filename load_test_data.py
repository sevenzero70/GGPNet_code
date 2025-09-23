"""
Notice:
     These functions are for the PanCollection dataset only
     PanCollection: https://github.com/liangjiandeng/PanCollection
"""

import cv2
import h5py
import torch
import numpy as np
import scipy.io as sio


def get_edge_mat(data):  # get high-frequency
    rs = np.zeros_like(data)
    if len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = data[:, :, i] - cv2.boxFilter(data[:, :, i], -1, (5, 5))
    else:
        rs = data - cv2.boxFilter(data, -1, (5, 5))
    return rs


def get_edge_h5py(data):  # get high-frequency
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


def load_mat(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8
    ms = torch.from_numpy(data['ms'] / 2047.0)  # HxWxC = 256x256x8
    ms = ms.permute(2, 0, 1)
    lms = torch.from_numpy(data['lms'] / 2047.0)  # HxWxC = 256x256x8
    lms = lms.permute(2, 0, 1)
    pan = torch.from_numpy(data['pan'] / 2047.0)   # HxW = 256x256
    return ms, lms, pan


def load_mat_with_hp(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8
    ms = torch.from_numpy(data['ms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms_hp = torch.from_numpy(get_edge_mat(data['ms'] / 2047.0)).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy(data['pan'] / 2047.0)  # HxW = 256x256
    pan_hp = torch.from_numpy(get_edge_mat(data['pan'] / 2047.0))   # HxW = 256x256
    return ms, ms_hp, pan, pan_hp


def load_h5py_u2(file_path):
    data = h5py.File(file_path)
    ms = data["ms"][...]  # W H C N
    ms = np.array(ms, dtype=np.float32) / 2047.
    ms = torch.from_numpy(ms)
    lms = data["lms"][...]  # W H C N
    lms = np.array(lms, dtype=np.float32) / 2047.
    lms = torch.from_numpy(lms)
    pan = data["pan"][...]  # W H C N
    pan = np.array(pan, dtype=np.float32) / 2047.
    pan = torch.from_numpy(pan)
    gt = data["gt"][...]  # W H C N
    gt = np.array(gt, dtype=np.float32) / 2047.
    gt = torch.from_numpy(gt)
    return ms, lms, pan, gt

def load_h5py(file_path):
    data = h5py.File(file_path)
    ms = data["ms"][...]  # W H C N
    ms = np.array(ms, dtype=np.float32) / 2047.
    ms = torch.from_numpy(ms)
    lms = data["lms"][...]  # W H C N
    lms = np.array(lms, dtype=np.float32) / 2047.
    lms = torch.from_numpy(lms)
    pan = data["pan"][...]  # W H C N
    pan = np.array(pan, dtype=np.float32) / 2047.
    pan = torch.from_numpy(pan)
    return ms, lms, pan


def load_h5py_with_hp(file_path):
    data = h5py.File(file_path)
    ms1 = data["ms"][...]
    ms = np.array(ms1, dtype=np.float32) / 2047.
    ms = torch.from_numpy(ms)
    ms_hp = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.  # NxHxWxC
    ms_hp = get_edge_h5py(ms_hp)
    ms_hp = torch.from_numpy(ms_hp).permute(0, 3, 1, 2)

    pan1 = data["pan"][...]  # W H C N
    pan = np.array(pan1, dtype=np.float32) / 2047.
    pan = torch.from_numpy(pan)
    pan_hp = np.array(pan1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.  # NxHxWx1
    pan_hp = np.squeeze(pan_hp, axis=3)  # NxHxW
    pan_hp = get_edge_h5py(pan_hp)  # NxHxW
    pan_hp = np.expand_dims(pan_hp, axis=3)  # NxHxWx1
    pan_hp = torch.from_numpy(pan_hp).permute(0, 3, 1, 2)

    gt1 = data["gt"][...]  # W H C N
    gt = np.array(gt1, dtype=np.float32) / 2047.
    gt = torch.from_numpy(gt)
    gt_hp = np.array(gt1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.  # NxHxWx1
    gt_hp = get_edge_h5py(gt_hp)  # NxHxW
    gt_hp = torch.from_numpy(gt_hp).permute(0, 3, 1, 2)
    return ms, ms_hp, pan, pan_hp, gt, gt_hp

def load_mat_return_ms_pan(ms_path, pan_path):
    #首先处理MST网络重建出来的数据，以其作为SR模型输入的ms
    mat_data = sio.loadmat(ms_path)
    mat_data = mat_data['pred'][...]# pred数据形状(20，64，64，4)# print(mat data.shape)
    mat_data = mat_data.transpose(0,3,1,2)# print(mat data.shape)#(20，4，64,64)ms = np.array(mat data, dtype=np.float32)ms = torch.from _numpy(ms)
    ms = np.array(mat_data, dtype=np.float32)
    ms = torch.from_numpy(ms)
    #随后开始从测试数据中提取pan图像用来指导SR过程pan data = h5py.File(pan path)
    pan_data = h5py.File(pan_path)
    pan = pan_data['pan']
    pan = np.array(pan, dtype=np.float32)/ 2047
    pan = torch.from_numpy(pan)
    # pan = F.interpolate(pan, size=(64,64),mode='bilinear', align_corners=False)
    gt = pan_data['gt']
    gt = np.array(gt, dtype=np.float32)/ 2047
    gt = torch.from_numpy(gt)
    return ms, pan, gt