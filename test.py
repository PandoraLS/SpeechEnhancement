# -*- coding: utf-8 -*-
# Author：lisen
# Date：20-1-5 下午9:53

import argparse
import os

import json5
import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer.tester import Tester
from util.utils import initialize_config

def main(config, resume):
    torch.manual_seed(config["seed"])  # both CPU and CUDA
    np.random.seed(config["seed"])


    test_dataloader = DataLoader(
        dataset=initialize_config(config["test_dataset"]),
        batch_size=1,
        num_workers=1,
    )
    
    generator = initialize_config(config["generator_model"])
    discriminator = initialize_config(config["discriminator_model"])

    generator_optimizer = torch.optim.Adam(
        params=generator.parameters(),
        lr=config["optimizer"]["G_lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )
    discriminator_optimizer = torch.optim.Adam(
        params=discriminator.parameters(),
        lr=config["optimizer"]["D_lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    additional_loss_function = initialize_config(config["additional_loss_function"])
    
    tester = Tester(
        config=config,
        resume=resume,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        additional_loss_function=additional_loss_function,
        test_dl=test_dataloader
    )

    tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UNetGAN")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.json).",default="config/train/20200105_lisen.json5")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume experiment from latest checkpoint.",default=True)
    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["config_path"] = args.configuration

    main(configuration, resume=args.resume)
