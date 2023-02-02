import os
import torch
import librosa
import torchaudio
import numpy as np
import streamlit as st
import librosa.display
import pyloudnorm as pyln
import matplotlib.pyplot as plt

from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade


st.set_page_config(page_title="aimless-splitter")
st.image("docs/aimless-logo-crop.png", use_column_width="always")

bundle = HDEMUCS_HIGH_MUSDB_PLUS
sample_rate = bundle.sample_rate
fade_overlap = 0.1


@st.experimental_singleton
def load_hdemucs():
    print("Loading pretrained HDEMUCS model...")
    model = bundle.get_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model


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


def process_file(file, model: torch.nn.Module, device: torch.device):
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
        x = torchaudio.functional.resample(sr, sample_rate)(x)

    st.subheader("Mix")
    x_numpy = x.numpy()
    plot_spectrogram(x_numpy, sample_rate=sample_rate)
    st.audio(x_numpy, sample_rate=sample_rate)

    waveform = x.to(device)

    # split into 10.0 sec chunks
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()  # normalization
    sources = separate_sources(
        model,
        waveform[None],
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


# load pretrained model
hdemucs = load_hdemucs()

# load audio
uploaded_file = st.file_uploader("Choose a file to demix.")

if uploaded_file is not None:
    # split with hdemucs
    hdemucs_sources = process_file(uploaded_file, hdemucs, "cuda:0")
