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


## Install the repository as a package

```sh
pip install git+https://yoyololicon:ACCESS_TOKEN@github.com/yoyololicon/mdx23-aim-playground
```
For the value of `ACCESS_TOKEN` please refer to [#24](https://github.com/yoyololicon/mdx23-aim-playground/issues/24#issuecomment-1420952853).


### Training Details

* Clean songs (no label noise) are hand-labeled and recorded in `data/lightning/label_noise.csv`
* A hybrid demux model is trained with negative SDR as the loss function
* Training occurs on random chunks and random stem combinations of the clean songs
* Training batches are augmented and processed using different random effects
* Due to all this randomization, validation is done also on the training dataset (no separate validation set)
* All details and hyperparameters are contained in `cfg/hdemucs.yaml`
* Model is trained for ~800 epochs (approx. 2 weeks on 4 GPUs)
* During the last ~200 epochs, the learning rate is reduced to 0.001, gradient accumulation is increased to 64, and the effect randomization chance is increased by a factor of 1.666 (e.g. 30% to 50% etc.)
