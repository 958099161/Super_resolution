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
    tp = 256 #args.patchSize  # 256
    ip = tp // scale  # 64
    # imgIn_list=np.zeros((10,ip,ip,3))
    # imgHR_list = np.zeros((10, tp, tp, 3))
    #
    # for i in range(10):
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    imgIn = imgIn[iy:iy + ip, ix:ix + ip, :]
    imgTar = imgTar[ty:ty + tp, tx:tx + tp, :]
        # imgIn_list[i,:,:,:]=imgIn
        # imgHR_list[i, :, :, :] = imgTar
    return imgIn, imgTar


class vedio_data(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.scale = 4 #args.scale
        apath = 'E:/ali_uku' #args.dataDir
        self.dirHR = os.path.join(apath, 'round1_train_label')
        # self.dirHR_refer = os.path.join(apath, 'refer_128x128')
        self.dirLR = os.path.join(apath, 'round1_train_input')
        # self.dirTar = os.path.join(apath, dirHR)
        self.fileList = os.listdir(self.dirLR)
        # self.file_pathlist_hr = []
        # for name in self.fileList:
        #     if name.endswith('_pic'):
        #         # print(name)
        #         list_pic_file = os.listdir(self.dirHR + "/" + name)
        #         for file_name in list_pic_file:
        #             for dir in os.listdir(os.path.join(self.dirHR, name) + "/" + file_name):
        #                 self.file_pathlist_hr.append(os.path.join(self.dirHR, name) + "/" + file_name + "/" + dir)
        self.file_pathlist_lr = []
        for name in os.listdir(self.dirLR):
            if name.endswith('_pic'):
                # print(name)
                list_pic_file = os.listdir(self.dirLR + "/" + name)
                for file_name in list_pic_file:
                    for dir in os.listdir(os.path.join(self.dirLR, name) + "/" + file_name):
                        self.file_pathlist_lr.append(os.path.join(self.dirLR, name) + "/" + file_name + "/" + dir)

        self.len = len(self.file_pathlist_lr)
        # E:/ali_uku\round1_train_label\youku_00000_00049_h_GT_pic/Youku_00000_h_GT/image001.bmp

    def __getitem__(self, idx):
        scale = self.scale
        nameLr, nameHr, name = self.getFileName(idx)
        imgHr = cv2.imread(nameHr)
        imgLr = cv2.imread(nameLr)
        # imgLR = cv2.resize(imgLR, (128, 128), interpolation=cv2.INTER_CUBIC)  # vdsr�ȷŴ�
        # imgHR_refer = cv2.imread(dirHR_refer)
        # if self.args.need_patch:
        # if self.args.need_patch:
        #     imgIn, imgTar = getPatch(imgIn, imgTar, self.args, 2)
        # # imgIn, imgTar = augment(imgIn, imgTar)
        # # return RGB_np2Tensor(imgIn, imgTar)
        imgLR, imgHr = getPatch(imgLr, imgHr, self.args, scale)

        imgLR = imgLR / 255.0
        imgLR = imgLR.transpose((2, 0, 1))

        imgHr = imgHr / 255.0
        imgHr = imgHr.transpose((2, 0, 1))

        return (imgLR, imgHr, name)

    def __len__(self):
        return self.len

    def getFileName(self, idx):
        ## E:\ali_uku\round1_train_input\youku_00000_00049_l_pic\Youku_00000_l/image001.bmp
        ## E:/ali_uku\round1_train_label\youku_00000_00049_h_GT_pic/Youku_00000_h_GT/image001.bmp
        lr_name = self.file_pathlist_lr[idx]
        l1 = os.path.split(lr_name)[1]
        l2 = os.path.split(lr_name)[0]

        l2_0 = l2.split("\\")[0]
        l2_1 = l2.split("\\")[1]  #
        l2_3 = l2.split("\\")[2]  #

        hr_name = self.dirHR + "\\" + l2_3.split("_l")[0] + "_h_GT_pic\\" + l2_3.split("/")[1].split("_l")[
            0] + "_h_GT" + "//" + l1  # + l2_1.split("_l")[0]

        return lr_name, hr_name, l1
