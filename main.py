import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from aimless.lightning.waveform import WaveformSeparator
from aimless.lightning.freq_mask import MaskPredictor


def cli_main():
    cli = LightningCLI(
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": DDPStrategy(find_unused_parameters=False),
            "log_every_n_steps": 1,
        }
    )


if __name__ == "__main__":
    cli_main()
