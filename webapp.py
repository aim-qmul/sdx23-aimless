import os
import yaml
import torch
import librosa
import argparse
import torchaudio
import numpy as np
import streamlit as st
import librosa.display
import pyloudnorm as pyln
import matplotlib.pyplot as plt

from pytorch_lightning.cli import LightningCLI

from importlib import import_module
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade

from lightning.waveform import WaveformSeparator
from lightning.freq_mask import MaskPredictor
from lightning.data import MUSDB

from utils import MDX_SOURCES, SDX_SOURCES


@st.experimental_singleton
def load_hdemucs():
    print("Loading pretrained HDEMUCS model...")
    model = bundle.get_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model


# @st.experimental_singleton
def load_checkpoint(config: str, ckpt_path: str):
    with open(config) as f:
        config = yaml.safe_load(f)

    model_configs = config["model"]

    # first build the base model
    base_model_configs = model_configs["init_args"]["model"]
    module_path, class_name = base_model_configs["class_path"].rsplit(".", 1)
    module = import_module(module_path)
    base_model = getattr(module, class_name)(**base_model_configs["init_args"])

    # load weights from checkpoint
    lightning_state = torch.load(ckpt_path, map_location="cpu")
    model_state = lightning_state["state_dict"]

    # parse out weights only for the base model
    base_model_state = {}
    for key, val in model_state.items():
        if "model" in key:
            base_model_state[key.replace("model.", "")] = val
    base_model.load_state_dict(base_model_state, strict=True)

    # use default source ordering from utils
    if config["model"]["init_args"]["use_sdx_targets"]:
        sources = SDX_SOURCES
    else:
        sources = MDX_SOURCES

    base_model.sources = sources
    base_model.eval()
    return base_model


def plot_spectrogram(y, *, sample_rate, figsize=(12, 3)):
    # Convert to mono
    if y.ndim > 1:
        y = y[0]

    fig = plt.figure(figsize=figsize)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max, top_db=120)
    img = librosa.display.specshow(D, y_axis="linear", x_axis="time", sr=sample_rate)
    fig.colorbar(img, format="%+2.f dB")
    st.pyplot(fig=fig, clear_figure=True)


def separate_sources(
    model: torch.nn.Module,
    mix: torch.Tensor,
    segment: float = 10.0,
    overlap: float = 0.1,
    device=None,
):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


def process_file(
    file,
    model: torch.nn.Module,
    device: torch.device,
    sample_rate: float,
):
    import tempfile
    import shutil
    from pathlib import Path

    # Cache file to disk so we can read it with ffmpeg
    with tempfile.NamedTemporaryFile("wb", suffix=Path(file.name).suffix) as f:
        shutil.copyfileobj(file, f)

        duration = librosa.get_duration(filename=f.name)
        num_frames = -1
        if duration > 60:
            st.write(f"File is {duration:.01f}s long. Loading the first 60s only.")
            sr = librosa.get_samplerate(f.name)
            num_frames = 60 * sr

        x, sr = torchaudio.load(f.name, num_frames=num_frames)

    # resample if needed
    if sr != sample_rate:
        x = torchaudio.functional.resample(x, sr, sample_rate)

    st.subheader("Mix")
    x_numpy = x.numpy()
    plot_spectrogram(x_numpy, sample_rate=sample_rate)
    st.audio(x_numpy, sample_rate=sample_rate)

    waveform = x.to(device)

    # If mono, make stereo
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    # Otherwise just take first 2 channels
    elif x.shape[0] > 2:
        x = x[:2]

    # split into 10.0 sec chunks
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization
    sources = separate_sources(
        model,
        waveform.unsqueeze(0),
        device=device,
    )[0]
    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)
    audios = dict(zip(sources_list, sources))

    for source, audio in audios.items():
        audio = audio.cpu().numpy()
        st.subheader(source.capitalize())
        plot_spectrogram(audio, sample_rate=sample_rate)
        st.audio(audio, sample_rate=sample_rate)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Path to YAML configuration.",
        default=None,
    )
    parser.add_argument(
        "--ckpt_path",
        help="Path to pretrained checkpoint with model weights.",
        default=None,
    )

    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently streamlit prevents the program from exiting normally
        # so we have to do a hard exit.
        os._exit(e.code)

    if args.config is None:
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        sample_rate = bundle.sample_rate
        model = load_hdemucs()  # load pretrained model
    else:
        sample_rate = 44100
        model = load_checkpoint(args.config, args.ckpt_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    st.set_page_config(page_title="aimless-splitter")
    st.image("docs/aimless-logo-crop.png", use_column_width="always")

    # load audio
    uploaded_file = st.file_uploader("Choose a file to demix.")

    if uploaded_file is not None:
        # split with model
        process_file(uploaded_file, model, device, sample_rate)
