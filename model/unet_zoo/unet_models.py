# -*- coding: utf-8 -*-
# Author：sen
# Date：2020/3/23 9:31

import torch.nn.functional as F

from .unet_parts import *
from .multi_scale import *

class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()
        
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.Conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.Conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.Conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.Conv5 = ConvBlock(ch_in=512, ch_out=1024)
        
        self.Up5 = UpConv(ch_in=1024, ch_out=512)
        self.Up_conv5 = ConvBlock(ch_in=1024, ch_out=512)
        
        self.Up4 = UpConv(ch_in=512, ch_out=256)
        self.Up_conv4 = ConvBlock(ch_in=512, ch_out=256)
        
        self.Up3 = UpConv(ch_in=128, ch_out=64)
        self.Up_conv3 = ConvBlock(ch_in=256, ch_out=128)
        
        self.Up2 = UpConv(ch_in=128, ch_out=64)
        self.Up_conv2 = ConvBlock(ch_in=128, ch_out=64)
        
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        
        x2 = self.MaxPool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.MaxPool(x2)
        x3 = self.Conv3(x3)
        
        x4 = self.MaxPool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.MaxPool(x4)
        x5 = self.Conv5(x5)
        
        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Conv_1x1(d2)
        
        return d1
    
# TODO 剩下的UNet网络
        