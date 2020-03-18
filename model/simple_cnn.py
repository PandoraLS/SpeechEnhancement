# -*- coding: utf-8 -*-
# Author：lisen
# Date：20-3-8 下午2:43

import torch
import torch.nn as nn


class base_cnn(nn.Module):
    def __init__(self):
        super(base_cnn, self).__init__()

        self.conv = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=64,
                              stride=32, padding=16)
        self.deconv = nn.ConvTranspose1d(in_channels=512, out_channels=1, kernel_size=64,
                                         stride=32, padding=16)

    def forward(self, x):
        batch_size = x.size(0)  # batchsize
        time_stamp = x.size(1)  # length of x
        # reshape to facilitate transform
        x = x.view(batch_size, 1, time_stamp)  # torch.size([batch_size, 1, timp_stamp])

        ft_ly = self.conv(x)  # torch.size([batch_size, 512, 512])
        output = self.deconv(ft_ly)
        output = output.view(batch_size, -1)

        return output


if __name__ == '__main__':
    # 默认情况下是使用gpu的方式
    inpt = torch.rand(2, 16384).cuda()
    Model = base_cnn().cuda()
    output = Model(inpt)
    print(output.shape)
