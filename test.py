#------------------
# Author luzhongshan
# Time2019/5/25 11:23
#------------------
import os
import argparse
import math
import cv2
import numpy as np
import torch
from model1 import RCAN
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
from tensorboardX import SummaryWriter



parser = argparse.ArgumentParser(description='Semantic aware super-resolution')
# ##########################################################
parser.add_argument('--model_choose', default='CARN', help='model directory')
parser.add_argument('--model_savepath', default='E:\lunwen\RCAN-pytorch-master\weight', help='dataset directory')
parser.add_argument('--model_name', default='RCAN_epoch_5_28_-2-28.pth', help='model directory')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size for training')
# ######################################################################
parser.add_argument('--result_SR_Dir', default='E:/ali_uku/round1_train_result', help='datasave directory')
parser.add_argument('--LR_Dir', default='test', help=' directory')
parser.add_argument('--dataDir', default='E:/ali_uku', help='dataset directory')

args = parser.parse_args()
from data.my_data import vedio_data
def get_dataset(args):
    data_train = vedio_data(args)
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batchSize, shuffle=False,num_workers=1)
    return dataloader

from utils import saveData
def test(args):
    my_model = RCAN(args)  # model.RDN()

    save = saveData(args)
    dataloader = get_dataset(args)
    my_model.cuda()
    # my_model.eval()
    model_path = os.path.join(args.model_savepath, args.model_name)
    # my_model.load_state_dict(torch.load(model_path))
    my_model = save.load_model(my_model, model_path)
    for i, (lr_in, name) in enumerate(dataloader):
        # _,_,w,h =lr_in.shape
        # out_img =torch.ze  切图
        # lr_in_ = lr_in.numpy()
        _, _, w, h = lr_in.shape
        # print(_,_,w,h)
        # out_img = np.zeros((1, 3, w, h))
        # in_img1 = np.zeros((1, 3, int(w / 3), int(h / 2)))
        # in_img2 = np.zeros((1, 3, int(w / 2), int(h / 2)))
        # for i in range(5):
        #     img_hr_out=np.zeros((3,w*4,h*4))
        #     for j in range(10):
        # img_hr_out=np.zeros((3,w*4,h*4))
        # z=0
        # in_img1 = np.zeros((6,3, 90,60))
        # for i_w in range(3):
        #     for i_h in range(2):
        #         in_img1[z,:,:,:] =lr_in[0,:,(i_w)*(int(w /3)):(i_w+1)*(int(w /3)),(i_h)*(int(h /8)):(i_h+1)*(int(h /8))]
        #         z=z+1
                # in_img1
                # in_img2 = lr_in[:, :, 0:w, int(h / 2):]
        # in_img1=torch.from_numpy(in_img1)
                # in_img2 = torch.from_numpy(in_img2)
        in_img1 = lr_in.cuda().float()#, volatile=False)
        in_img1 = my_model(in_img1)
        in_img1 = in_img1[0]
        img_hr_out = in_img1.cpu().data.numpy()
        # z=0
        # for i_w in range(3):
        #     for i_h in range(2):
        #         img_hr_out[:,(i_w)*(int(w /3))*4:(i_w+1)*(int(w /3))*4,4*(i_h)*(int(h /8)):4*(i_h+1)*(int(h /8))] = in_img1[z,:,:,:]
        #         z=z+1
        img_hr_out = img_hr_out.transpose((1, 2, 0))
        img_hr_out = img_hr_out
        img_hr_out = np.ceil(img_hr_out * 256)

        img_hr_out[img_hr_out>255]=255
        img_hr_out[img_hr_out <0] = 0
        # img_hr_out1 = np.zeros((1080,1920,3))

        out_i = i // 100
        i_i = i % 100
        # img_hr_out1 =cv2.imread(args.result_SR_Dir + "/sr/"+str(out_i) + '/%05d_sr.bmp' % (i_i))
        #拼图片
        # img_hr_out1[360:2*360,0:960,:]=img_hr_out

        if not os.path.exists(args.result_SR_Dir + "/sr/"+str(out_i)):
            os.mkdir(args.result_SR_Dir + "/sr/"+str(out_i))
        cv2.imwrite(args.result_SR_Dir + "/sr/"+str(out_i) + '/%05d_sr.bmp' % (i_i), img_hr_out)


if __name__ == '__main__':
    test(args)
