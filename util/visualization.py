# -*- coding: utf-8 -*-
# Author：sen
# Date：2020/3/14 12:30

from torch.utils.tensorboard import SummaryWriter

def writer(logs_dir):
    return SummaryWriter(log_dir=logs_dir, max_queue=5, flush_secs=30)