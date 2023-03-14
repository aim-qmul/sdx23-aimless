import torchaudio
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
import random
import torch
import soundfile as sf
import numpy as np
from resampy import resample
from typing import Optional, Callable

__all__ = ["SpeechNoise"]


class SpeechNoise(Dataset):
    sr: int = 44100

    def __init__(
        self,
        speech_root: str,
        noise_root: str,
        seq_duration: float = 6.0,
        samples_per_track: int = 64,
        least_overlap_ratio: float = 0.5,
        snr_sampler: Optional[torch.distributions.Distribution] = None,
        mono: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        speech_root = Path(speech_root)
        noise_root = Path(noise_root)
        speech_files = list(speech_root.glob("**/*.wav")) + list(
            speech_root.glob("**/*.flac")
        )
        noise_files = list(Path(noise_root).glob("**/*.wav")) + list(
            Path(noise_root).glob("**/*.flac")
        )

        speech_track_frames = []
        speech_track_sr = []
        for x in tqdm(speech_files):
            info = torchaudio.info(x)
            speech_track_frames.append(info.num_frames)
            speech_track_sr.append(info.sample_rate)

        noise_track_frames = []
        noise_track_sr = []
        for x in tqdm(noise_files):
            info = torchaudio.info(x)
            noise_track_frames.append(info.num_frames)
            noise_track_sr.append(info.sample_rate)

        self.speech_files = speech_files
        self.noise_files = noise_files
        self.speech_track_frames = speech_track_frames
        self.noise_track_frames = noise_track_frames
        self.speech_track_sr = speech_track_sr
        self.noise_track_sr = noise_track_sr

        self.seq_duration = seq_duration
        self.samples_per_track = samples_per_track
        self.segment = int(self.seq_duration * self.sr)
        self.least_overlap_ratio = least_overlap_ratio
        self.least_overlap_segment = int(least_overlap_ratio * self.segment)
        self.transform = transform
        self.snr_sampler = snr_sampler
        self.mono = mono

        self._size = len(self.noise_files) * self.samples_per_track

    def __len__(self):
        return self._size

    def _get_random_track_idx(self):
        return random.randrange(len(self.tracks))

    def _get_random_start(self, length):
        return random.randrange(length - self.segment + 1)

    def __getitem__(self, index):
        track_idx = index // self.samples_per_track
        noise_sr = self.noise_track_sr[track_idx]
        noise_resample_ratio = self.sr / noise_sr
        noise_file = self.noise_files[track_idx]
        pos_start = int(
            self._get_random_start(
                int(self.noise_track_frames[track_idx] * noise_resample_ratio)
            )
            / noise_resample_ratio
        )
        frames = int(self.seq_duration * noise_sr)
        noise, _ = sf.read(
            noise_file, start=pos_start, frames=frames, fill_value=0, always_2d=True
        )
        if noise_sr != self.sr:
            noise = resample(noise, noise_sr, self.sr, axis=0)
            if noise.shape[0] < self.segment:
                noise = np.pad(
                    noise, ((0, self.segment - noise.shape[0]), (0, 0)), "constant"
                )
            else:
                noise = noise[: self.segment]

        if self.mono:
            noise = noise.mean(axis=1, keepdims=True)
        else:
            noise = np.broadcast_to(noise, (noise.shape[0], 2))

        # get a random speech file
        speech_idx = random.randint(0, len(self.speech_files) - 1)
        speech_file = self.speech_files[speech_idx]
        speech_sr = self.speech_track_sr[speech_idx]
        speech_resample_ratio = self.sr / speech_sr
        speech_resampled_length = int(
            self.speech_track_frames[speech_idx] * speech_resample_ratio
        )

        if speech_resampled_length < self.least_overlap_segment:
            speech, _ = sf.read(speech_file, always_2d=True)
            if speech_sr != self.sr:
                speech = resample(speech, speech_sr, self.sr, axis=0)
                speech_resampled_length = speech.shape[0]

            if self.mono:
                speech = speech.mean(axis=1, keepdims=True)
            else:
                speech = np.broadcast_to(speech, (speech.shape[0], 2))

            speech_energy = np.sum(speech**2)
            pos = random.randint(0, self.segment - speech_resampled_length)

            padded_speech = np.zeros_like(noise)
            padded_speech[pos : pos + speech_resampled_length] = speech
            speech = padded_speech
        else:
            pos = random.randint(
                self.least_overlap_segment - speech_resampled_length,
                self.segment - self.least_overlap_segment,
            )
            if pos < 0:
                pos_start = int(-pos / speech_resample_ratio)
                frames = int(
                    min(self.segment, (speech_resampled_length + pos))
                    / speech_resample_ratio
                )
            else:
                pos_start = 0
                frames = int(
                    min(speech_resampled_length, self.segment - pos)
                    / speech_resample_ratio
                )

            speech, _ = sf.read(
                speech_file,
                start=pos_start,
                frames=frames,
                fill_value=0,
                always_2d=True,
            )
            if speech_sr != self.sr:
                speech = resample(speech, speech_sr, self.sr, axis=0)

            if self.mono:
                speech = speech.mean(axis=1, keepdims=True)
            else:
                speech = np.broadcast_to(speech, (speech.shape[0], 2))

            speech_energy = np.sum(speech**2)

            padded_speech = np.zeros_like(noise)
            pos = max(0, pos)
            padded_speech[pos : pos + speech.shape[0]] = speech
            speech = padded_speech

        speech = torch.from_numpy(speech.T).float()
        noise = torch.from_numpy(noise.T).float()

        if self.snr_sampler is not None:
            snr = self.snr_sampler.sample()
            # scale noise to have the desired SNR
            noise_energy = noise.pow(2).sum() + 1e-8
            noise = noise * torch.sqrt(speech_energy / noise_energy) * 10 ** (-snr / 10)

        stems = torch.stack([speech, noise], dim=0)
        mix = speech + noise

        if self.transform is not None:
            mix, stems = self.transform((mix, stems))

        return mix, stems
