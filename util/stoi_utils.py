# -*- coding: utf-8 -*-
# Author：sen
# Date：2020/3/23 14:35
import torch
import torch.nn as nn
import numpy as np
from numpy import std, vstack, hstack, argsort, argmax, array, hanning, savez, real, imag
import scipy.signal
import torch.nn.functional as F

EPS = np.finfo(float).eps


def pearsonr(x, y, return_batch=False):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y

    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    mean_x = torch.mean(x, dim=1, keepdim=True)
    mean_y = torch.mean(y, dim=1, keepdim=True)
    xm = x - mean_x
    ym = y - mean_y
    r_num = torch.sum(xm * ym, dim=1, keepdim=True)
    xm_2_sum = torch.sqrt(torch.sum(torch.pow(xm, 2), dim=1, keepdim=True) + EPS)
    ym_2_sum = torch.sqrt(torch.sum(torch.pow(ym, 2), dim=1, keepdim=True) + EPS)

    r_den = xm_2_sum * ym_2_sum + EPS
    r_val = r_num / r_den
    return r_val if return_batch else torch.mean(r_val)


class FourierTransformer(nn.Module):
    """
    用神经网络来模拟STFT变换
    """

    def __init__(self, windowsize=1024, hop=64, window='hann', requires_grad=False):
        super().__init__()

        self.windowsize = windowsize
        self.hop = hop

        win = scipy.signal.get_window(window, windowsize)  # 窗的类型+窗的采样点
        f = np.fft.fft(np.eye(windowsize))  # 这里的f实际上表示的是权重W
        f *= win  # 权重乘以窗
        f = vstack((real(f[:int(windowsize / 2 + 1), :]), imag(f[:int(windowsize / 2 + 1), :])))
        # fft后，生成的数据长度有windowsize个采样点，然而傅里叶变换是对称的，为防止数据冗余，只取前半部分
        # 先取f的前半部分，vstack 将 real 和 imag 叠加组合， 这样矩阵的大小和原来是一样的
        f = f[:, None, :]  # 增加一个维度， 这里的f就是kernel
        self.ft = torch.nn.Parameter(torch.FloatTensor(f), requires_grad=requires_grad)

    def forward(self, x):
        # x  torch.size([batch_size, 1, 16384])
        # bt = x.size(0)  # batchsize
        # T = x.size(2)  # x数据的时间采样点
        # x = x.view(bt, 1, T)  # 16 x 1 x 16000
        tx = F.conv1d(x, self.ft, stride=self.hop)
        amp = torch.sqrt(
            tx[:, :int(self.windowsize / 2) + 1, :] ** 2 + tx[:, int(self.windowsize / 2) + 1:, :] ** 2 + EPS)
        ph = torch.atan2(tx[:, int(self.windowsize / 2) + 1:, :], tx[:, :int(self.windowsize / 2) + 1, :])  # 弧度制
        amp = torch.transpose(amp, dim0=1, dim1=2).contiguous()
        ph = torch.transpose(ph, dim0=1, dim1=2).contiguous()
        return amp, ph  # batch x num_frame x num_freq


class STOICalculator(nn.Module):
    """
    将STOI计算指标融入到损失函数中，该实现来自于Paper：
    Fu, Szu-Wei, et al. "End-to-end waveform utterance enhancement for direct evaluation metrics optimization
    by fully convolutional neural networks." IEEE/ACM Transactions on Audio, Speech, and Language Processing
    26.9 (2018): 1570-1584.
    """

    def __init__(self, samplerate=16000, windowsize=1024, hop=64, window='hann',
                 num_bands=15, low_center_freq=150, seg_len=96, seg_hop=1,
                 requires_grad=False):
        super().__init__()
        self.seg_len = seg_len
        self.seg_hop = seg_hop
        self.num_bands = num_bands
        self.stft = FourierTransformer(windowsize, hop, window, requires_grad)
        oct_m, _ = self._thirdoct(samplerate, windowsize, num_bands, low_center_freq)
        oct_m = oct_m.T  # num_freq * num_bands
        self.oct_m = torch.nn.Parameter(torch.FloatTensor(oct_m), requires_grad=requires_grad)
        seg_eye = np.eye(seg_len)
        self.seg_eye = torch.nn.Parameter(torch.FloatTensor(seg_eye[:, None, :]), requires_grad=requires_grad)

    def forward(self, deg, clean):
        # stft
        x_amp, x_ph = self.stft(deg)  # batch * num_frame * num_freq
        y_amp, y_ph = self.stft(clean)

        batchsize, nframe, nfreq = x_amp.size(0), x_amp.size(1), x_amp.size(2)

        # 1/3 octave band grouping
        x_amp = x_amp.view(-1, nfreq)
        x_amp = x_amp.mm(self.oct_m)
        y_amp = y_amp.view(-1, nfreq)
        y_amp = y_amp.mm(self.oct_m)  # batchsize * nframe * num_bands

        # segment v3
        x_amp = x_amp.view(batchsize, nframe, self.num_bands)
        x_amp = torch.transpose(x_amp, dim0=1, dim1=2).contiguous().view(batchsize * self.num_bands, nframe)
        y_amp = y_amp.view(batchsize, nframe, self.num_bands)
        y_amp = torch.transpose(y_amp, dim0=1, dim1=2).contiguous().view(batchsize * self.num_bands, nframe)
        x_amp = F.conv1d(x_amp[:, None, :], self.seg_eye,
                         stride=self.seg_hop)  # batchsize * num_bands * seg_len * num_seg
        y_amp = F.conv1d(y_amp[:, None, :], self.seg_eye, stride=self.seg_hop)

        # normalization v2
        num_seg = x_amp.size(2)
        x_amp = x_amp.view(batchsize, self.num_bands, self.seg_len, num_seg)
        y_amp = y_amp.view(batchsize, self.num_bands, self.seg_len, num_seg)

        x_ss = torch.sum(x_amp ** 2, dim=2, keepdim=True) + EPS
        y_ss = torch.sum(y_amp ** 2, dim=2, keepdim=True)
        alpha = torch.sqrt(y_ss / x_ss + EPS)

        alpha = alpha.expand(-1, -1, self.seg_len, -1)
        x_amp = x_amp * alpha  # v4, fix bug

        x_amp = torch.transpose(x_amp, dim0=2, dim1=3).contiguous().view(-1, self.seg_len)
        y_amp = torch.transpose(y_amp, dim0=2, dim1=3).contiguous().view(-1, self.seg_len)

        stoi = pearsonr(x_amp, y_amp)
        return stoi

    def _thirdoct(self, fs, N_fft, numBands, mn):
        step = float(fs) / N_fft
        f = np.array([i * step for i in range(0, N_fft // 2 + 1)])
        k = np.arange(numBands)
        cf = 2 ** (k / 3) * mn
        fl = np.sqrt(cf * 2 ** ((k - 1.) / 3) * mn + EPS)
        fr = np.sqrt(cf * 2 ** ((k + 1.) / 3) * mn + EPS)
        A = [[0 for j in range(len(f))] for i in range(numBands)]

        for i in range(len(cf)):
            fl_i = np.argmin((f - fl[i]) ** 2)
            fr_i = np.argmin((f - fr[i]) ** 2)
            for j in range(fl_i, fr_i):
                A[i][j] = 1
        A = np.array(A)
        rnk = np.sum(A, axis=1)

        for i in range(len(rnk) - 2, 0, -1):
            if rnk[i + 1] >= rnk[i] and rnk[i + 1] != 0:
                numBands = i + 1
                break

        A = A[0: numBands + 1, :]
        cf = cf[0: numBands + 1]
        return A, cf
