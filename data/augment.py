import torch
import random
import torchaudio
import numpy as np
import scipy.stats
import scipy.signal
import pyloudnorm as pyln
from typing import List, Tuple

from pedalboard import (
    Pedalboard,
    Gain,
    Chorus,
    Reverb,
    Compressor,
    Phaser,
    Delay,
    Distortion,
    Limiter,
)


__all__ = [
    "RandomSwapLR",
    "RandomGain",
    "RandomFlipPhase",
    "LimitAug",
    "CPUBase",
    "RandomParametricEQ",
    "RandomStereoWidener",
    "RandomVolumeAutomation",
    "RandomPedalboardCompressor",
    "RandomPedalboardDelay",
    "RandomPedalboardChorus",
    "RandomPedalboardDistortion",
    "RandomPedalboardPhaser",
    "RandomPedalboardReverb",
    "RandomPedalboardLimiter",
    "RandomSoxReverb",
    "LoudnessNormalize",
    "RandomPan",
    "Mono2Stereo",
]


class CPUBase(object):
    def __init__(self, p: float = 1.0):
        assert 0 <= p <= 1, "invalid probability value"
        self.p = p

    def __call__(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (Tuple[torch.Tensor, torch.Tensor]): (mixture, stems) where
                mixture : (Num_channels, L)
                stems: (Num_sources, Num_channels, L)
        Return:
            Tuple[torch.Tensor, torch.Tensor]: (mixture, stems) where
                mixture: (Num_channels, L)
                stems: (Num_sources, Num_channels, L)
        """
        mixture, stems = x
        stems_fx = []
        for stem_idx in np.arange(stems.shape[0]):
            stem = stems[stem_idx, ...]
            if np.random.rand() < self.p:
                stems_fx.append(self._transform(stem))
            else:
                stems_fx.append(stem)
        stems = torch.stack(stems_fx, dim=0)
        mixture = stems.sum(0)
        return (mixture, stems)

    def _transform(self, stems: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RandomSwapLR(CPUBase):
    def __init__(self, p=0.5) -> None:
        super().__init__(p=p)

    def _transform(self, stems: torch.Tensor):
        return torch.flip(stems, [0])


class RandomGain(CPUBase):
    def __init__(self, low=0.25, high=1.25, **kwargs) -> None:
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def _transform(self, stems):
        gain = np.random.rand() * (self.high - self.low) + self.low
        return stems * gain


class RandomFlipPhase(RandomSwapLR):
    def __init__(self, p=0.5) -> None:
        super().__init__(p=p)

    def _transform(self, stems: torch.Tensor):
        return -stems


def db2linear(x):
    return 10 ** (x / 20)


class LimitAug(CPUBase):
    def __init__(
        self,
        target_lufs_mean=-10.887,
        target_lufs_std=1.191,
        target_loudnorm_lufs=-14.0,
        max_release_ms=200.0,
        min_release_ms=30.0,
        sample_rate=44100,
        **kwargs,
    ) -> None:
        """
        Args:
            target_lufs_mean (float): mean of target LUFS. default: -10.887 (corresponding to the statistics of musdb-L)
            target_lufs_std (float): std of target LUFS. default: 1.191 (corresponding to the statistics of musdb-L)
            target_loudnorm_lufs (float): target LUFS after loudnorm. default: -14.0
            max_release_ms (float): max release time of limiter. default: 200.0
            min_release_ms (float): min release time of limiter. default: 30.0
            sample_rate (int): sample rate of audio. default: 44100
        """
        super().__init__(**kwargs)
        self.target_lufs_sampler = torch.distributions.Normal(
            target_lufs_mean, target_lufs_std
        )
        self.target_loudnorm_lufs = target_loudnorm_lufs
        self.sample_rate = sample_rate
        self.board = Pedalboard([Gain(0), Limiter(threshold_db=0.0, release_ms=100.0)])
        self.limiter_release_sampler = torch.distributions.Uniform(
            min_release_ms, max_release_ms
        )
        self.meter = pyln.Meter(sample_rate)

    def __call__(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.rand() > self.p:
            return x
        mixture, stems = x
        mixture_np = mixture.numpy().T
        loudness = self.meter.integrated_loudness(mixture_np)
        target_lufs = self.target_lufs_sampler.sample().item()
        self.board[1].release_ms = self.limiter_release_sampler.sample().item()

        if np.isinf(loudness):
            aug_gain = 0.0
        else:
            aug_gain = target_lufs - loudness
        self.board[0].gain_db = aug_gain

        new_mixture_np = self.board(mixture_np, self.sample_rate)
        after_loudness = self.meter.integrated_loudness(new_mixture_np)

        if not np.isinf(after_loudness):
            target_gain = self.target_loudnorm_lufs - after_loudness
            new_mixture_np *= db2linear(target_gain)

        new_mixture = torch.tensor(new_mixture_np.T, dtype=mixture.dtype)
        # apply element-wise gain to stems
        stems *= new_mixture.abs() / mixture.abs().add(1e-8)
        return (new_mixture, stems)


def loguniform(low=0, high=1):
    return scipy.stats.loguniform.rvs(low, high)


def rand(low=0, high=1):
    return (torch.rand(1).numpy()[0] * (high - low)) + low


def randint(low=0, high=1):
    return torch.randint(low, high + 1, (1,)).numpy()[0]


def biqaud(
    gain_db: float,
    cutoff_freq: float,
    q_factor: float,
    sample_rate: float,
    filter_type: str,
):
    """Use design parameters to generate coeffieicnets for a specific filter type.
    Args:
        gain_db (float): Shelving filter gain in dB.
        cutoff_freq (float): Cutoff frequency in Hz.
        q_factor (float): Q factor.
        sample_rate (float): Sample rate in Hz.
        filter_type (str): Filter type.
            One of "low_shelf", "high_shelf", or "peaking"
    Returns:
        b (np.ndarray): Numerator filter coefficients stored as [b0, b1, b2]
        a (np.ndarray): Denominator filter coefficients stored as [a0, a1, a2]
    """

    A = 10 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * (cutoff_freq / sample_rate)
    alpha = np.sin(w0) / (2.0 * q_factor)

    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)

    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    else:
        pass
        # raise ValueError(f"Invalid filter_type: {filter_type}.")

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0

    return b, a


def parametric_eq(
    x: np.ndarray,
    sample_rate: float,
    low_shelf_gain_db: float = 0.0,
    low_shelf_cutoff_freq: float = 80.0,
    low_shelf_q_factor: float = 0.707,
    band_gains_db: List[float] = [0.0],
    band_cutoff_freqs: List[float] = [300.0],
    band_q_factors: List[float] = [0.707],
    high_shelf_gain_db: float = 0.0,
    high_shelf_cutoff_freq: float = 1000.0,
    high_shelf_q_factor: float = 0.707,
    dtype=np.float32,
):
    """Multiband parametric EQ.

    Low-shelf -> Band 1 -> ... -> Band N -> High-shelf

    Args:

    """
    assert (
        len(band_gains_db) == len(band_cutoff_freqs) == len(band_q_factors)
    )  # must define for all bands

    # -------- apply low-shelf filter --------
    b, a = biqaud(
        low_shelf_gain_db,
        low_shelf_cutoff_freq,
        low_shelf_q_factor,
        sample_rate,
        "low_shelf",
    )
    x = scipy.signal.lfilter(b, a, x)

    # -------- apply peaking filters --------
    for gain_db, cutoff_freq, q_factor in zip(
        band_gains_db, band_cutoff_freqs, band_q_factors
    ):
        b, a = biqaud(
            gain_db,
            cutoff_freq,
            q_factor,
            sample_rate,
            "peaking",
        )
        x = scipy.signal.lfilter(b, a, x)

    # -------- apply high-shelf filter --------
    b, a = biqaud(
        high_shelf_gain_db,
        high_shelf_cutoff_freq,
        high_shelf_q_factor,
        sample_rate,
        "high_shelf",
    )
    sos5 = np.concatenate((b, a))
    x = scipy.signal.lfilter(b, a, x)

    return x.astype(dtype)


class RandomParametricEQ(CPUBase):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        num_bands: int = 3,
        min_gain_db: float = -6.0,
        max_gain_db: float = +6.0,
        min_cutoff_freq: float = 1000.0,
        max_cutoff_freq: float = 10000.0,
        min_q_factor: float = 0.1,
        max_q_factor: float = 4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.num_bands = num_bands
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        self.min_q_factor = min_q_factor
        self.max_q_factor = max_q_factor

    def _transform(self, x: torch.Tensor):
        """
        Args:
            x: (torch.Tensor): Array of audio samples with shape (chs, seq_leq).
                The filter will be applied the final dimension, and by default the same
                filter will be applied to all channels.
        """
        low_shelf_gain_db = rand(self.min_gain_db, self.max_gain_db)
        low_shelf_cutoff_freq = loguniform(20.0, 200.0)
        low_shelf_q_factor = rand(self.min_q_factor, self.max_q_factor)

        high_shelf_gain_db = rand(self.min_gain_db, self.max_gain_db)
        high_shelf_cutoff_freq = loguniform(8000.0, 16000.0)
        high_shelf_q_factor = rand(self.min_q_factor, self.max_q_factor)

        band_gain_dbs = []
        band_cutoff_freqs = []
        band_q_factors = []
        for _ in range(self.num_bands):
            band_gain_dbs.append(rand(self.min_gain_db, self.max_gain_db))
            band_cutoff_freqs.append(
                loguniform(self.min_cutoff_freq, self.max_cutoff_freq)
            )
            band_q_factors.append(rand(self.min_q_factor, self.max_q_factor))

        y = parametric_eq(
            x.numpy(),
            self.sample_rate,
            low_shelf_gain_db=low_shelf_gain_db,
            low_shelf_cutoff_freq=low_shelf_cutoff_freq,
            low_shelf_q_factor=low_shelf_q_factor,
            band_gains_db=band_gain_dbs,
            band_cutoff_freqs=band_cutoff_freqs,
            band_q_factors=band_q_factors,
            high_shelf_gain_db=high_shelf_gain_db,
            high_shelf_cutoff_freq=high_shelf_cutoff_freq,
            high_shelf_q_factor=high_shelf_q_factor,
        )

        return torch.from_numpy(y)


def stereo_widener(x: torch.Tensor, width: torch.Tensor):
    sqrt2 = np.sqrt(2)

    left = x[0, ...]
    right = x[1, ...]

    mid = (left + right) / sqrt2
    side = (left - right) / sqrt2

    # amplify mid and side signal seperately:
    mid *= 2 * (1 - width)
    side *= 2 * width

    left = (mid + side) / sqrt2
    right = (mid - side) / sqrt2

    x = torch.stack((left, right), dim=0)

    return x


class RandomStereoWidener(CPUBase):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_width: float = 0.0,
        max_width: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_width = min_width
        self.max_width = max_width

    def _transform(self, x: torch.Tensor):
        width = rand(self.min_width, self.max_width)
        return stereo_widener(x, width)


class RandomVolumeAutomation(CPUBase):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_segment_seconds: float = 3.0,
        min_gain_db: float = -6.0,
        max_gain_db: float = 6.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_segment_seconds = min_segment_seconds
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def _transform(self, x: torch.Tensor):
        gain_db = torch.zeros(x.shape[-1]).type_as(x)

        seconds = x.shape[-1] / self.sample_rate
        max_num_segments = max(1, int(seconds // self.min_segment_seconds))

        num_segments = randint(1, max_num_segments)
        segment_lengths = (
            x.shape[-1]
            * np.random.dirichlet(
                [rand(0, 10) for _ in range(num_segments)], 1
            )  # TODO(cm): this can crash training
        ).astype("int")[0]

        segment_lengths = np.maximum(segment_lengths, 1)

        # check the sum is equal to the length of the signal
        diff = segment_lengths.sum() - x.shape[-1]
        if diff < 0:
            segment_lengths[-1] -= diff
        elif diff > 0:
            for idx in range(num_segments):
                if segment_lengths[idx] > diff + 1:
                    segment_lengths[idx] -= diff
                    break

        samples_filled = 0
        start_gain_db = 0
        for idx in range(num_segments):
            segment_samples = segment_lengths[idx]
            if idx != 0:
                start_gain_db = end_gain_db

            # sample random end gain
            end_gain_db = rand(self.min_gain_db, self.max_gain_db)
            fade = torch.linspace(start_gain_db, end_gain_db, steps=segment_samples)
            gain_db[samples_filled : samples_filled + segment_samples] = fade
            samples_filled = samples_filled + segment_samples

        # print(gain_db)
        x *= 10 ** (gain_db / 20.0)
        return x


class RandomPedalboardCompressor(CPUBase):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_threshold_db: float = -42.0,
        max_threshold_db: float = -6.0,
        min_ratio: float = 1.5,
        max_ratio: float = 4.0,
        min_attack_ms: float = 1.0,
        max_attack_ms: float = 50.0,
        min_release_ms: float = 10.0,
        max_release_ms: float = 250.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_threshold_db = min_threshold_db
        self.max_threshold_db = max_threshold_db
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_attack_ms = min_attack_ms
        self.max_attack_ms = max_attack_ms
        self.min_release_ms = min_release_ms
        self.max_release_ms = max_release_ms

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        threshold_db = rand(self.min_threshold_db, self.max_threshold_db)
        ratio = rand(self.min_ratio, self.max_ratio)
        attack_ms = rand(self.min_attack_ms, self.max_attack_ms)
        release_ms = rand(self.min_release_ms, self.max_release_ms)

        board.append(
            Compressor(
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
            )
        )

        # process audio using the pedalboard
        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class RandomPedalboardDelay(CPUBase):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_delay_seconds: float = 0.1,
        max_delay_sconds: float = 1.0,
        min_feedback: float = 0.05,
        max_feedback: float = 0.6,
        min_mix: float = 0.0,
        max_mix: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_delay_seconds = min_delay_seconds
        self.max_delay_seconds = max_delay_sconds
        self.min_feedback = min_feedback
        self.max_feedback = max_feedback
        self.min_mix = min_mix
        self.max_mix = max_mix

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        delay_seconds = loguniform(self.min_delay_seconds, self.max_delay_seconds)
        feedback = rand(self.min_feedback, self.max_feedback)
        mix = rand(self.min_mix, self.max_mix)
        board.append(Delay(delay_seconds=delay_seconds, feedback=feedback, mix=mix))
        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class RandomPedalboardChorus(CPUBase):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_rate_hz: float = 0.25,
        max_rate_hz: float = 4.0,
        min_depth: float = 0.0,
        max_depth: float = 0.6,
        min_centre_delay_ms: float = 5.0,
        max_centre_delay_ms: float = 10.0,
        min_feedback: float = 0.1,
        max_feedback: float = 0.6,
        min_mix: float = 0.1,
        max_mix: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_rate_hz = min_rate_hz
        self.max_rate_hz = max_rate_hz
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_centre_delay_ms = min_centre_delay_ms
        self.max_centre_delay_ms = max_centre_delay_ms
        self.min_feedback = min_feedback
        self.max_feedback = max_feedback
        self.min_mix = min_mix
        self.max_mix = max_mix

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        rate_hz = rand(self.min_rate_hz, self.max_rate_hz)
        depth = rand(self.min_depth, self.max_depth)
        centre_delay_ms = rand(self.min_centre_delay_ms, self.max_centre_delay_ms)
        feedback = rand(self.min_feedback, self.max_feedback)
        mix = rand(self.min_mix, self.max_mix)
        board.append(
            Chorus(
                rate_hz=rate_hz,
                depth=depth,
                centre_delay_ms=centre_delay_ms,
                feedback=feedback,
                mix=mix,
            )
        )
        # process audio using the pedalboard
        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class RandomPedalboardPhaser(CPUBase):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_rate_hz: float = 0.25,
        max_rate_hz: float = 5.0,
        min_depth: float = 0.1,
        max_depth: float = 0.6,
        min_centre_frequency_hz: float = 200.0,
        max_centre_frequency_hz: float = 600.0,
        min_feedback: float = 0.1,
        max_feedback: float = 0.6,
        min_mix: float = 0.1,
        max_mix: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_rate_hz = min_rate_hz
        self.max_rate_hz = max_rate_hz
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_centre_frequency_hz = min_centre_frequency_hz
        self.max_centre_frequency_hz = max_centre_frequency_hz
        self.min_feedback = min_feedback
        self.max_feedback = max_feedback
        self.min_mix = min_mix
        self.max_mix = max_mix

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        rate_hz = rand(self.min_rate_hz, self.max_rate_hz)
        depth = rand(self.min_depth, self.max_depth)
        centre_frequency_hz = rand(
            self.min_centre_frequency_hz, self.min_centre_frequency_hz
        )
        feedback = rand(self.min_feedback, self.max_feedback)
        mix = rand(self.min_mix, self.max_mix)
        board.append(
            Phaser(
                rate_hz=rate_hz,
                depth=depth,
                centre_frequency_hz=centre_frequency_hz,
                feedback=feedback,
                mix=mix,
            )
        )
        # process audio using the pedalboard
        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class RandomPedalboardLimiter(CPUBase):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_threshold_db: float = -32.0,
        max_threshold_db: float = -6.0,
        min_release_ms: float = 10.0,
        max_release_ms: float = 300.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_threshold_db = min_threshold_db
        self.max_threshold_db = max_threshold_db
        self.min_release_ms = min_release_ms
        self.max_release_ms = max_release_ms

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        threshold_db = rand(self.min_threshold_db, self.max_threshold_db)
        release_ms = rand(self.min_release_ms, self.max_release_ms)
        board.append(
            Limiter(
                threshold_db=threshold_db,
                release_ms=release_ms,
            )
        )
        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class RandomPedalboardDistortion(CPUBase):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_drive_db: float = -20.0,
        max_drive_db: float = 12.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_drive_db = min_drive_db
        self.max_drive_db = max_drive_db

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        drive_db = rand(self.min_drive_db, self.max_drive_db)
        board.append(Distortion(drive_db=drive_db))
        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class RandomSoxReverb(CPUBase):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_reverberance: float = 10.0,
        max_reverberance: float = 100.0,
        min_high_freq_damping: float = 0.0,
        max_high_freq_damping: float = 100.0,
        min_wet_dry: float = 0.0,
        max_wet_dry: float = 1.0,
        min_room_scale: float = 5.0,
        max_room_scale: float = 100.0,
        min_stereo_depth: float = 20.0,
        max_stereo_depth: float = 100.0,
        min_pre_delay: float = 0.0,
        max_pre_delay: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_reverberance = min_reverberance
        self.max_reverberance = max_reverberance
        self.min_high_freq_damping = min_high_freq_damping
        self.max_high_freq_damping = max_high_freq_damping
        self.min_wet_dry = min_wet_dry
        self.max_wet_dry = max_wet_dry
        self.min_room_scale = min_room_scale
        self.max_room_scale = max_room_scale
        self.min_stereo_depth = min_stereo_depth
        self.max_stereo_depth = max_stereo_depth
        self.min_pre_delay = min_pre_delay
        self.max_pre_delay = max_pre_delay

    def _transform(self, x: torch.Tensor):
        reverberance = rand(self.min_reverberance, self.max_reverberance)
        high_freq_damping = rand(self.min_high_freq_damping, self.max_high_freq_damping)
        room_scale = rand(self.min_room_scale, self.max_room_scale)
        stereo_depth = rand(self.min_stereo_depth, self.max_stereo_depth)
        wet_dry = rand(self.min_wet_dry, self.max_wet_dry)
        pre_delay = rand(self.min_pre_delay, self.max_pre_delay)

        effects = [
            [
                "reverb",
                f"{reverberance}",
                f"{high_freq_damping}",
                f"{room_scale}",
                f"{stereo_depth}",
                f"{pre_delay}",
                "--wet-only",
            ]
        ]
        y, _ = torchaudio.sox_effects.apply_effects_tensor(
            x, self.sample_rate, effects, channels_first=True
        )

        # manual wet/dry mix
        return (x * (1 - wet_dry)) + (y * wet_dry)


class RandomPedalboardReverb(CPUBase):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        min_room_size: float = 0.0,
        max_room_size: float = 1.0,
        min_damping: float = 0.0,
        max_damping: float = 1.0,
        min_wet_dry: float = 0.0,
        max_wet_dry: float = 0.7,
        min_width: float = 0.0,
        max_width: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.min_wet_dry = min_wet_dry
        self.max_wet_dry = max_wet_dry
        self.min_width = min_width
        self.max_width = max_width

    def _transform(self, x: torch.Tensor):
        board = Pedalboard()
        room_size = rand(self.min_room_size, self.max_room_size)
        damping = rand(self.min_damping, self.max_damping)
        wet_dry = rand(self.min_wet_dry, self.max_wet_dry)
        width = rand(self.min_width, self.max_width)

        board.append(
            Reverb(
                room_size=room_size,
                damping=damping,
                wet_level=wet_dry,
                dry_level=(1 - wet_dry),
                width=width,
            )
        )

        return torch.from_numpy(board(x.numpy(), self.sample_rate))


class LoudnessNormalize(CPUBase):
    def __init__(
        self,
        sample_rate: float = 44100.0,
        target_lufs_db: float = -32.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(sample_rate)
        self.target_lufs_db = target_lufs_db

    def _transform(self, x: torch.Tensor):
        x_lufs_db = self.meter.integrated_loudness(x.permute(1, 0).numpy())
        delta_lufs_db = torch.tensor([self.target_lufs_db - x_lufs_db]).float()
        gain_lin = 10.0 ** (delta_lufs_db.clamp(-120, 40.0) / 20.0)
        return gain_lin * x


class Mono2Stereo(CPUBase):
    def __init__(self) -> None:
        super().__init__(p=1.0)

    def _transform(self, x: torch.Tensor):
        assert x.ndim == 2 and x.shape[0] == 1, "x must be mono"
        return torch.cat([x, x], dim=0)


class RandomPan(CPUBase):
    def __init__(
        self,
        min_pan: float = -1.0,
        max_pan: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.min_pan = min_pan
        self.max_pan = max_pan

    def _transform(self, x: torch.Tensor):
        """Constant power panning"""
        assert x.ndim == 2 and x.shape[0] == 2, "x must be stereo"
        theta = rand(self.min_pan, self.max_pan) * np.pi / 4
        x = x * 0.707  # normalize to prevent clipping
        left_x, right_x = x[0], x[1]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        left_x = left_x * (cos_theta - sin_theta)
        right_x = right_x * (cos_theta + sin_theta)
        return torch.stack([left_x, right_x], dim=0)
