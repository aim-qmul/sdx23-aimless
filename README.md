<div align="center">

<img width="400px" src="docs/aimless-logo-crop.svg">

</div>

AIMLESS (Artificial Intelligence and Music League for Effective Source Separation) is a special interest group in audio source separation at C4DM, consisting of PhD students from the AIM CDT program.
This repository contains our training code for the [SDX23 Sound Demixing Challenge](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023).


## Quick Start

The conda environment we used for training is described in `environment.yml`.
The below commands should be runnable if you're using QMUL EECS servers.
If you want to run it on your local machine, change the `root` param in the config to where you downloaded the MUSDB18-HQ dataset.

### Frequency-masking-based model


```commandline
python main.py fit --config cfg/xumx.yaml
```

### Raw-waveform-based model


```commandline
python main.py fit --config cfg/demucs.yaml
```

## Install the repository as a package

This step is required if you want to test our submission repositories (see section [Reproduce the winning submission](#reproduce-the-winning-submission)) locally.
```sh
pip install git+https://github.com/aim-qmul/sdx23-aimless
```

## Reproduce the winning submission

### CDX Leaderboard A, submission ID 220319

This section describes how to reproduce the [best perform model](https://gitlab.aicrowd.com/yoyololicon/cdx-submissions/-/issues/90) we used on CDX leaderboard A.
The submission consists of one HDemucs predicting all the targets and one BandSplitRNN predicitng the music from the mixture.

To train the HDemucs:
```commandline
python main.py fit --config cfg/cdx_a/hdemucs.yaml --data.init_args.root /DNR_DATASET_ROOT/dnr_v2/
```
Remember to change `/DNR_DATASET_ROOT/dnr_v2/` to your download location of [Divide and Remaster (DnR) dataset](https://zenodo.org/record/6949108).

To train the BandSplitRNN:
```commandline
python main.py fit --config cfg/cdx_a/bandsplit_rnn.yaml --data.init_args.root /DNR_DATASET_ROOT/dnr_v2/
```

We trained the models with no more than 4 GPUs, depending on the resources we had at the time.

After training, please go to our [submission repository](https://gitlab.aicrowd.com/yoyololicon/cdx-submissions/).
Then, copy the last checkpoint of HDemucs (usually located at `lightning_logs/version_**/checkpoints/last.ckpt`) to `my_submission/lightning_logs/hdemucs-64-sdr/checkpoints/last.ckpt` in the submission repository.
Similarly, copy the last checkpoint of BandSplitRNN to `my_submission/lightning_logs/bandsplitRNN-music/checkpoints/last.ckpt`.
After these steps, you have reproduced our submission!

The inference procedure in our submission repository is a bit complex.
Briefly speaking, the HDemucs predicts the targets independently for each channels of the stereo mixture, plus, the average (the mid) and the difference (the side) of the two channels.
The stereo separated sources are made from a linear combination of these mono predictions.
The separated music from the BandSplitRNN is enhanced by Wiener Filtering, and the final music predictions is the average from the two models.

### MDX Leaderboard A (Label Noise), submission ID 220426

This section describes how to reproduce the [best perform model](https://gitlab.aicrowd.com/yoyololicon/mdx23-submissions/-/issues/76) we used on MDX leaderboard A.

Firstly, we manually inspected the [label noise dataset](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23/dataset_files)(thanks @mhrice for the hard work!) and labeled the clean songs (no label noise).
The labels are recorded in `data/lightning/label_noise.csv`.
Then, a HDemucs was trained only on the clean labels with the following settings:

* negative SDR as the loss function
* Training occurs on random chunks and random stem combinations of the clean songs
* Training batches are augmented and processed using different random effects
* Due to all this randomization, validation is done also on the training dataset (no separate validation set)

To reproduce the training:
```commandline
python main.py fit --config cfg/mdx_a/hdemucs.yaml --data.init_args.root /DATASET_ROOT/
```
Remember to place the label noise data under `/DATASET_ROOT/train/`.

Other details:
* Model is trained for ~800 epochs (approx. 2 weeks on 4 RTX A50000)
* During the last ~200 epochs, the learning rate is reduced to 0.001, gradient accumulation is increased to 64, and the effect randomization chance is increased by a factor of 1.666 (e.g. 30% to 50% etc.)

After training, please go to our [submission repository](https://gitlab.aicrowd.com/yoyololicon/mdx23-submissions/).
Then, copy the checkpoint to `my_submission/acc64_4devices_lr0001_e1213_last.ckpt` in the submission repository.
After these steps, you have reproduced our submission!


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



