import os

import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.strategies import DDPStrategy


def cli_main():
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("medium")
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    cli = LightningCLI(
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": DDPStrategy(find_unused_parameters=False),
            "log_every_n_steps": 1,
        }
    )


if __name__ == "__main__":
    cli_main()
