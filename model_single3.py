#------------------
# Author luzhongshan
# Time2019/5/24 22:26
#------------------
#------------------
# Author luzhongshan
# Time2019/5/24 17:40
#------------------

from torch import nn
import torch

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):                 # 2层
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):   # 64  20  16
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]      # 20
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)


class RCAN(nn.Module):
    def __init__(self, args):
        super(RCAN, self).__init__()
        scale = 4 #args.scale
        num_features = 64 #args.num_features
        num_rg =10# args.num_rg
        num_rcab =20 # args.num_rcab
        reduction =16 # args.reduction

        self.sf = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.rgs1 = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(5)])

        self.rgs2 = nn.Sequential(*[RG(64, num_rcab, reduction) for _ in range(5)])

        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        self.sub =nn.PixelShuffle(2)
        self.relu =nn.ReLU()
        self.conv2 = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)
        self.conv_basic16_64 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv_basic = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(80, 64, kernel_size=1, padding=1)
    def forward(self, x):
        x = self.sf(x)
        # residual = x
        residual = self.upscale(x)  # 64
        x = self.upscale(x)
        # x = self.sub(x)        # 第一次放大
        residual1=x    #16

        # x1 = self.relu(x)
        # x1=self.conv_basic16_64(x)
        x = self.rgs2(x)
        # x1 = self.conv1(x1)
        x =torch.cat((x ,residual ,residual1),1)
        # x1 = self.relu(x1)
        x =self.conv_basic(x)

        x = self.upscale(x)
        # x1=self.relu(x1)            #添加
        # x = self.upscale(x)
        x = self.conv2(x)
        return x
