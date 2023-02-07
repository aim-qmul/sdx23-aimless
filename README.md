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

* `aimless`: package root, which can be imported for submission.
  * `loss`: loss functions.
    * `freq.*`: loss functions for frequency-domain models .
    * `time.*`: loss functions for time-domain models.
  * `augment`: data augmentations that are better on GPU.
  * `lightning`: all lightning modules.
    * `waveform.WaveformSeparator`: trainer for time-domain models.
    * `freq_mask.MaskPredictor`: trainer for frequency-domain models.
  * `models`: your custom models.
* `cfg`: all config files.
* `data`: 
  * `dataset`: custom pytorch datasets.
  * `lightning`: all lightning data modules.
  * `augment`: data augmentations that are better on CPU.

## Streamlit

Split song in the browser with pretrained Hybrid Demucs. 

``` streamlit run scripts/webapp.py ```

Then open [http://localhost:8501/](http://localhost:8501/) in your browser. 