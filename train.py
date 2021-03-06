# -*- coding: utf-8 -*-
# Author：sen
# Date：2020/3/22 15:47

import argparse
import os
import json5
import torch
import numpy as np
from torch.utils.data import DataLoader
from util.utils import initialize_config
from trainer.trainer import GeneralTrainer


def main(config, resume):
    
    torch.manual_seed(int(config["seed"]))  # both CPU and CUDA
    np.random.seed(config["seed"])

    train_dataloader = DataLoader(
        dataset=initialize_config(config["train_dataset"]),
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        shuffle=config["train_dataloader"]["shuffle"],
        pin_memory=config["train_dataloader"]["pin_memory"]  # Very small data set False
    )

    validation_dataloader = DataLoader(
        dataset=initialize_config(config["validation_dataset"]),
        batch_size=1,
        num_workers=1
    )

    model = initialize_config(config["model"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    loss_function = initialize_config(config["loss_function"])

    trainer = GeneralTrainer(
        config=config,
        resume=resume,
        model=model,
        optim=optimizer,
        loss_fucntion=loss_function,
        train_dl=train_dataloader,
        validation_dl=validation_dataloader
    )

    trainer.train()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="SimpleCNN")
    # parser.add_argument("-C", "--configuration", required=True, type=str, default='config/simple_cnn.json5',
    #                     help="Configuration (*.json).")
    # parser.add_argument("-R", "--resume", action="store_true", default=False,
    #                     help="Resume experiment from latest checkpoint.")
    # args = parser.parse_args()
    
    config_path = "config/20200323_simple_cnn.json5"
    
    configuration = json5.load(open(config_path))
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(config_path))
    configuration["config_path"] = config_path

    main(configuration, resume=False)
