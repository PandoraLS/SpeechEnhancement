# -*- coding: utf-8 -*-
# Author：lisen
# Date：20-3-8 下午4:23
import torch
import numpy as np
import json5
from pathlib import Path
from util import visualization
from util.utils import prepare_empty_dir

class BaseTrainer:
    def __init__(self, config, resume, model, optimizer, loss_function):
        self.n_gpu = torch.cuda.device_count()
        self.device = self._prepare_device(self.n_gpu, cudnn_deterministic=config["cudnn_deterministic"])

        self.model = model.to(self.device)

        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.n_gpu)))

        self.optimizer = optimizer

        self.loss_function = loss_function

        # Trainer
        self.epochs = config["trainer"]["epochs"]
        self.save_checkpoint_interval = config["trainer"]["save_checkpoint_interval"]
        self.validation_config = config["trainer"]["validation"]
        self.validation_interval = self.validation_config["validation"]
        self.find_max = self.validation_config["find_max"]
        self.validation_custom_config = self.validation_config["custom"]
        
        self.start_epoch = 1
        self.best_score = -np.inf if self.find_max else np.inf
        self.root_dir = Path(config["root_dir"]).expanduser().absolute() / config["experiment_name"] # 这一步是什么意思
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.logs_dir = self.root_dir / "logs"
        prepare_empty_dir([self.checkpoints_dir,self.logs_dir],resume=resume)
        
        self.writer = visualization.writer(self.logs_dir.as_posix())
        self.writer.add_text(
            tag="Configuration",
            text_string=f"<pre> \n{json5.dumps(config, indent=4, sort_keys=False)}  \n</pre>",
            global_step=1
        )
        
        if resume:self._resume_checkpoint()
        
        
        
    def _resume_checkpoint(self):
        pass