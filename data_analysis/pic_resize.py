#------------------
# Author luzhongshan
# Time2019/5/27 19:23
#------------------
import math
def get_psnr(y,x):     #行  列  通道
    im = y
    im2 = x
    # print (im.shape,im.dtype)
#图像的行数
    height = im.shape[0]
#图像的列数
    width = im.shape[1]

#提取R通道
    r = im[:,:,0]
#提取g通道
    g = im[:,:,1]
#提取b通道
    b = im[:,:,2]
#打印g通道数组
#print (g)
#图像1,2各自分量相减，然后做平方；
    R = im[:,:,0]-im2[:,:,0]
    G = im[:,:,1]-im2[:,:,1]
    B = im[:,:,2]-im2[:,:,2]
#做平方
    mser = R*R
    mseg = G*G
    mseb = B*B
#三个分量差的平方求和
    SUM = mser.sum() + mseg.sum() + mseb.sum()
    MSE = SUM / (height * width * 3)
    MSE=math.fabs(MSE)
    PSNR = 10*math.log((255.0*255.0/((MSE)*1.0)),10)
    return PSNR

import os
import numpy as np
import cv2
str_big="image001.bmp"
str_small ="00000_sr.bmp"
big = cv2.imread(str_big)
big1=np.transpose(big,(2,0,1))
big_tosmall =cv2.resize(big,(480,270),interpolation=cv2.INTER_AREA)  #cv2.INTER_AREA 43
print(big1.shape)
small = cv2.imread(str_small)
# small1=np.transpose(small,(2,0,1))
# print(small1.shape)
# print("....................")
# c,w,h =big1.shape
# neww =int(w//4)
# newh=int(h//4)
# print(neww,newh)
# out_img = np.zeros((3,neww,newh))
# for i in range(neww):
#     for j in range(newh):
#         out_img[:,i,j] = big1[:,4*i,4*j]
# out_img1=np.transpose(out_img,(1,2,0))
print(get_psnr(big,small))