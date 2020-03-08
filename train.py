import argparse
import os

import json5
import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer.trainer import Trainer
from util.utils import initialize_config

def main(config, resume):
    torch.manual_seed(config["seed"])  # both CPU and CUDA
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

    trainer = Trainer(
        config=config,
        resume=resume,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        additional_loss_function=additional_loss_function,
        train_dl=train_dataloader,
        validation_dl=validation_dataloader,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UNetGAN")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.json).",default="config/train/20200105_lisen.json5")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume experiment from latest checkpoint.",default=True)
    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["config_path"] = args.configuration

    main(configuration, resume=args.resume)
