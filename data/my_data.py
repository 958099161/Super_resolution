# ------------------
# Author luzhongshan
# Time2019/5/25 11:25
# ------------------
import os
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data


def augment(imgIn, imgTar):
    if random.random() < 0.3:
        imgIn = imgIn[:, ::-1, :]
        imgTar = imgTar[:, ::-1, :]
    if random.random() < 0.3:
        imgIn = imgIn[::-1, :, :]
        imgTar = imgTar[::-1, :, :]
    return imgIn, imgTar


def getPatch(imgIn, imgTar, args, scale):
    (ih, iw, c) = imgIn.shape
    (th, tw) = (scale * ih, scale * iw)
    tp = args.patchSize  # 256
    ip = tp // scale  # 64
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    imgIn = imgIn[iy:iy + ip, ix:ix + ip, :]
    imgTar = imgTar[ty:ty + tp, tx:tx + tp, :]
    return imgIn, imgTar


class vedio_data(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.scale = 4 #args.scale
        apath = args.dataDir
        # self.dirHR = os.path.join(apath, args.HR_Dir)
        self.dirLR = os.path.join(apath, args.LR_Dir)
        self.fileList = os.listdir(self.dirLR)
        self.file_pathlist_lr = []
        for name in os.listdir(self.dirLR):
            if name.endswith('l_pic'):
                list_pic_file = os.listdir(self.dirLR + "/" + name)
                for file_name in list_pic_file:
                    for dir in os.listdir(os.path.join(self.dirLR, name) + "/" + file_name):
                        self.file_pathlist_lr.append(os.path.join(self.dirLR, name) + "/" + file_name + "/" + dir)

        self.len = len(self.file_pathlist_lr)
        # E:/ali_uku\round1_train_label\youku_00000_00049_h_GT_pic/Youku_00000_h_GT/image001.bmp

    def __getitem__(self, idx):

        nameLr, name = self.getFileName(idx)
        imgLr = cv2.imread(nameLr)
        imgLr = imgLr / 256.0
        imgLr = imgLr.transpose((2, 0, 1))
        return imgLr, name

    def __len__(self):
        return self.len

    def getFileName(self, idx):
        ## E:\ali_uku\round1_train_input\youku_00000_00049_l_pic\Youku_00000_l/image001.bmp
        ## E:/ali_uku\round1_train_label\youku_00000_00049_h_GT_pic/Youku_00000_h_GT/image001.bmp
        lr_name = self.file_pathlist_lr[idx]
        # l1 = os.path.split(lr_name)[1]
        # l2 = os.path.split(lr_name)[0]
        # l2_0 = l2.split("\\")[0]
        # l2_1 = l2.split("\\")[1]  #
        # l2_3 = l2.split("\\")[2]  #
        # hr_name = self.dirHR + "\\"   + l2_3.split("_l")[0] +"_h_GT_pic\\"+ l2_3.split("/")[1].split("_l")[0]+"_h_GT" + "//" + l1 #+ l2_1.split("_l")[0]
        return lr_name, lr_name
