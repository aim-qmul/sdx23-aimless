<div align="center">

<img width="400px" src="docs/aimless-logo-crop.svg">

</div>

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

## Streamlit

Split song in the browser with pretrained Hybrid Demucs. 

```
CUDA_VISIBLE_DEVICES=0 python -m streamlit run webapp.py \
--server.fileWatcherType none \ 
-- \ 
--logdir /import/c4dm-datasets-ext/sdx-2023/logs-cjs/lightning_logs/version_0

Or, use your own pretrained model. Just pass the top level log directory.

```
CUDA_VISIBLE_DEVICES=0 python -m streamlit run webapp.py \
--server.fileWatcherType none \
-- \
--logdir /import/c4dm-datasets-ext/sdx-2023/logs-cjs/lightning_logs/version_0
```

Then forward port `8501` and open [http://localhost:8501/](http://localhost:8501/) in your browser. 