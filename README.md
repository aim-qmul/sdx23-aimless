<div align="center">

<img width="400px" src="docs/aimless-logo-crop.svg">

</div>

## Quick Start

The conda environment we used for training is described in `environment.yml`.
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

## Reproduce the winning submission

This section describes how to reproduce the models we used in our [winning submission](https://gitlab.aicrowd.com/yoyololicon/cdx-submissions/-/issues/90) on CDX leaderboard A.
The submission consists of one HDemucs predicting all the targets and one BandSplitRNN predicitng the music from the mixture.

To train the HDemucs:
```commandline
python main.py fit --config cfg/cdx_a/hdemucs.yaml --data.init_args.root /DNR_DATASET_ROOT/dnr_v2/
```
Remember to change `/DNR_DATASET_ROOT/dnr_v2/` to your download location of [Divide and Remaster (DnR) dataset](https://zenodo.org/record/6949108).

To train the BandSplitRNN:
```commandline
python main.py fit --config cfg/cdx_a/bandsplit_rnn.yaml --data.init_args.root DNR_DATASET_ROOT/dnr_v2/
```

We trained the models with no more than 4 GPUs, depending on the resources we had at the time.

After training, please go to our [submission repository](https://gitlab.aicrowd.com/yoyololicon/cdx-submissions/) and checkout the tag `submission-hdemucs+rnn-music-99k`.
Then, copy the last checkpoint of HDemucs (usually located at `lightning_logs/version_**/checkpoints/last.ckpt`) to `my_submission/lightning_logs/hdemucs-64-sdr/checkpoints/last.ckpt` in the submission repository.
Similarly, copy the last checkpoint of BandSplitRNN to `my_submission/lightning_logs/bandsplitRNN-music/checkpoints/last.ckpt`.
After these steps, you have reproduced our submission!

The inference procedure in our submission repository is a bit complex.
Briefly speaking, the Hdemucs predicts the targets independently for each channels of the stereo mixture, plus, the average (the mid) and the difference (the side) of the two channels.
The stereo separated sources are made from a linear combination of these mono predictions.
The separated music from the BandSplitRNN is enhanced by Wiener Filtering, and the final music predictions is the average from the two models.

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
