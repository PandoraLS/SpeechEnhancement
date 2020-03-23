# -*- coding: utf-8 -*-
# Author：lisen
# Date：20-3-7 下午4:44
import os
import glob
import time
import numpy as np
import importlib
from sphfile import SPHFile
from pypesq import pesq
from pystoi.stoi import stoi
import librosa


class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        
        <Something...>
        
        print(f"Finished in {timer.duration()} seconds.")
    """

    def __init__(self):
        self.start_time = time.time()

    def duratioin(self):
        return time.time() - self.start_time


def initialize_config(module_cfg):
    """
    根据配置项，动态加载对应的模块， 并将参数传入模块内部的制定函数
    eg. 配置文件如下:
        module_cfg = {
            "module": "models.unet", 
            "main": "UNet",
            "args": {...}
        }
    1. 加载 type 参数对应的模块
    2. 调用(实例化)模块内部对应 main 参数的函数(类)
    3. 再调用(实例化)时将 args 参数输入函数(类)
    
    
    :param module_cfg: 配置信息， 参见json文件
    :return: 实例化后的函数(类)
    """
    module = importlib.import_module(module_cfg["module"])
    return getattr(module, module_cfg["main"])(**module_cfg["args"])


def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.
    :param dirs (list): directors list 
    :param resume (bool):  是否继续试验，默认是False
    :return: 
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


def set_requires_grad(nets, requires_grad=False):
    """
    :param nets: list of networks
    :param requires_grad: 
    :return: 
    """
    if not isinstance(nets, list):
        nets = [nets]

    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)


def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    return pesq(clean_signal, noisy_signal, sr)


def create_data_list(cleanRoot, noisyRoot):
    """
    将文件夹中的文件路径整理,并生成dataset.txt文件
    :param cleanRoot:  干净语音的路径
    :param noisyRoot:  带噪语音的路径
    Notes:
            the format of the waveform dataset.txt is as follows.
            In list file:
            <abs path of noisy wav 1><space><abs path of the clean wav 1>
            <abs path of noisy wav 2><space><abs path of the clean wav 2>
            ...
            <abs path of noisy wav n><space><abs path of the clean wav n>

            e.g.
            In "dataset.txt":
            /home/dog/train/noisy/a.wav /home/dog/train/clean/a.wav
            /home/dog/train/noisy/b.wav /home/dog/train/clean/b.wav
            ...
            /home/dog/train/noisy/x.wav /home/dog/train/clean/x.wav
    """
    count = 0
    txtFile = "dataset.txt"
    txtFileList = []

    for root, dirs, files in sorted(os.walk(noisyRoot)):
        for name in sorted(files):
            file = os.path.join(root, name)
            cleanFile = os.path.join(cleanRoot, name)
            txtFileList.append(file + " " + cleanFile)
            count += 1
            # print(file)
    if not os.path.exists(txtFile):
        os.mknod(txtFile)
    f_dst = open(txtFile, 'w')
    for i in txtFileList:
        f_dst.write(i)
        f_dst.write("\n")
    f_dst.close()
    print("文件总数量:", count)


def sample_fixed_length_data_aligned(data_a, data_b, sample_length):
    """
    sampling with fixed-length from tow waveform
    :param data_a:  numpy形式的audio_a
    :param data_b:  numpy格式的audio_b
    :param sample_length: 裁剪长度
    :return: 随机从语音中选取一段长度为sample_length的,用于输入到神经网络中
    """
    assert len(data_a) == len(data_b), "Inconsistent data length, unable to sampling."
    assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    frames_total = len(data_a)
    start = np.random.randint(frames_total - sample_length + 1)
    end = start + sample_length
    return data_a[start:end], data_b[start:end]


def timit_trans():
    # 下载的TIMIT可能无法直接使用,需要用此脚本转换一下
    path = '/home/lisen/uestc/Research/Dataset/TIMIT/TRAIN/*/*/*.WAV'
    sph_files = glob.glob(path)
    print(len(sph_files), "train utterences")
    for i in sph_files:
        sph = SPHFile(i)
        sph.write_wav(filename=i.replace(".WAV", "_.wav"))  # _不能删除
        os.remove(i)
    path = '/home/lisen/uestc/Research/Dataset/TIMIT/TEST/*/*/*.WAV'
    sph_files_test = glob.glob(path)
    print(len(sph_files_test), "test utterences")
    for i in sph_files_test:
        sph = SPHFile(i)
        sph.write_wav(filename=i.replace(".WAV", "_.wav"))  # _不能删除
        os.remove(i)
    print("Completed")


if __name__ == '__main__':
    pass
    # cleanRoot = '/home/lisen/uestc/Research/Dataset/ToyData/train_clean'
    # noisyRoot = '/home/lisen/uestc/Research/Dataset/ToyData/val_babble_0db'
    # create_data_list(cleanRoot, noisyRoot)

    # timit_trans()
    # caculate_length(cleanRoot)
