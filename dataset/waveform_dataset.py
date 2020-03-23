# -*- coding: utf-8 -*-
# Author：lisen
# Date：20-3-8 下午3:33
import os
import librosa
import numpy as np
from torch.utils.data import Dataset
from util.utils import sample_fixed_length_data_aligned


class WaveformDataset(Dataset):
    def __init__(self, dataset, limit=None, offset=0, sample_length=16384, train=True):
        """
        Construct training dataset
        Args:
            dataset (str): the path of dataset list，see "Notes"
            limit (int): the limit of dataset
            offset (int): the offset of dataset
            sample_length(int): the model only support fixed-length input in training, this parameter specify the input size of the model.
            train(bool): In training, the model need fixed-length input; in test, the model need fully-length input.

        Notes:
            the format of the waveform dataset is as follows.

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

        Return:
            mixture signals, clean signals, filename
        """
        super(WaveformDataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)))]
        dataset_list = dataset_list[offset:]  # 取前offset行数据
        if limit:
            dataset_list = dataset_list[:limit]
        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_path, clean_path = self.dataset_list[item].split(" ")
        name = os.path.splitext(os.path.basename(noisy_path))[0]

        noisy, _ = librosa.load(noisy_path, sr=None)
        clean, _ = librosa.load(clean_path, sr=None)

        if self.train:
            noisy, clean = sample_fixed_length_data_aligned(noisy, clean, self.sample_length)
            noisy = 2 * (noisy - np.min(noisy)) / (np.max(noisy) - np.min(noisy)) - 1  # 将音频数据归一化到[-1.0,1.0]
            clean = 2 * (clean - np.min(clean)) / (np.max(clean) - np.min(clean)) - 1
        
        # 语音的shape(1, length) 
        return noisy.reshape(1, -1), clean.reshape(1, -1), name
    
if __name__ == '__main__':
    from pprint import pprint
    wave_data = WaveformDataset(
        dataset="train.txt",
        sample_length=16384,
        train=True
    )
    dataset_list_length = wave_data.__len__()
    dataset_list = wave_data.dataset_list
    pprint(dataset_list_length)
    pprint(dataset_list)
    
    
    
