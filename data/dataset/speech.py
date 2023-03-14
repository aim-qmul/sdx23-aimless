import torchaudio
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
import random
import torch
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
        speech_files = list(Path(speech_root).glob("**/*.wav"))
        noise_files = list(Path(noise_root).glob("**/*.wav"))

        speech_track_lengths = []
        for x in tqdm(speech_files):
            assert torchaudio.info(x).sample_rate == SpeechNoise.sr
            speech_track_lengths.append(torchaudio.info(x).num_frames)

        noise_track_lengths = []
        for x in tqdm(noise_files):
            assert torchaudio.info(x).sample_rate == SpeechNoise.sr
            noise_track_lengths.append(torchaudio.info(x).num_frames)

        self.speech_files = speech_files
        self.noise_files = noise_files
        self.speech_track_lengths = speech_track_lengths
        self.noise_track_lengths = noise_track_lengths

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
        noise_file, chunk_start = self.noise_files[track_idx], self._get_random_start(
            self.track_lengths[track_idx]
        )

        noise, _ = torchaudio.load(
            noise_file, frame_offset=chunk_start, num_frames=self.segment
        )
        if self.mono:
            noise = noise.mean(dim=0, keepdim=True)
        else:
            noise = noise.broadcast_to(2, noise.shape[1])

        # get a random speech file
        speech_idx = random.randint(0, len(self.speech_files) - 1)
        speech_file = self.speech_files[speech_idx]
        speech_length = self.speech_track_lengths[speech_idx]

        if speech_length < self.least_overlap_segment:
            speech, _ = torchaudio.load(speech_file)
            if self.mono:
                speech = speech.mean(dim=0, keepdim=True)
            else:
                speech = speech.broadcast_to(2, speech.shape[1])

            pos = random.randint(0, self.segment - speech_length)

            padded_speech = torch.zeros_like(noise)
            padded_speech[:, pos : pos + speech_length] = speech
            speech = padded_speech
        else:
            pos = random.randint(
                self.least_overlap_segment - speech_length,
                self.segment - self.least_overlap_segment,
            )
            if pos < 0:
                frame_offset = -pos
                num_frames = speech_length + pos
            else:
                frame_offset = 0
                num_frames = min(speech_length, self.segment - pos)

            speech, _ = torchaudio.load(
                speech_file, frame_offset=frame_offset, num_frames=num_frames
            )
            if self.mono:
                speech = speech.mean(dim=0, keepdim=True)
            else:
                speech = speech.broadcast_to(2, speech.shape[1])

            padded_speech = torch.zeros_like(noise)
            pos = max(0, pos)
            padded_speech[:, pos : pos + speech.shape[1]] = speech
            speech = padded_speech

        snr = self.snr_sampler.sample()
        # scale noise to have the desired SNR
        noise_energy = noise.pow(2).sum()
        speech_energy = speech.pow(2).sum()
        noise = noise * torch.sqrt(speech_energy / noise_energy) * 10 ** (-snr / 10)

        stems = torch.cat([speech, noise], dim=0)
        mix = speech + noise
        return mix, stems
