# AIM-LESS CODEBASE

## Quick Start

You can build a conda environment using `environment.yml`.
The below commands should be runnable if you're using EECS servers.
If you want to run it on your local machine, change the `root` param in the config to where you downloaded the MUSDB18-HQ dataset.

### Frequency-masking-based model


```commandline
python main.py fit --config cfg/xumx.yaml
```

### Raw-waveform-based model


```commandline
python main.py fit --config cfg/demucs.yaml
```

## Structure

* `loss`: loss functions.
  * `freq.*`: loss functions for frequency-domain models .
  * `time.*`: loss functions for time-domain models.
* `augment`: data augmentation.
  * `cpu.*`: better to run on CPU.
  * `cuda.*`: better to run on GPU.
* `data`: any datasets.
* `lightning`: all lightning modules.
  * `waveform.WaveformSeparator`: trainer for time-domain models.
  * `freq_mask.MaskPredictor`: trainer for frequency-domain models.
* `models`: your custom models.
* `cfg`: all config files.