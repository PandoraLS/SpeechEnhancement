# -*- coding: utf-8 -*-
# Author：lisen
# Date：20-3-8 下午2:43

import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()

        self.conv = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=64,
                              stride=32, padding=16)
        self.deconv = nn.ConvTranspose1d(in_channels=512, out_channels=1, kernel_size=64,
                                         stride=32, padding=16)

    def forward(self, x):
        # x  torch.size([batch_size, 1, 16384])
        batch_size = x.size(0)  # batchsize
        time_stamp = x.size(2)  # length of x

        ft_ly = self.conv(x)  # torch.size([batch_size, 512, 512])
        output = self.deconv(ft_ly)
        return output


if __name__ == '__main__':
    # 默认情况下是使用gpu的方式
    inpt = torch.rand(2, 1, 16384).cuda()
    Model = BaseCNN().cuda()
    output = Model(inpt)
    print(output.shape)
