import os
import torch
import torchaudio
from torchvision.transforms import Compose, RandomApply
from augment.cpu import (
    RandomPebalboardReverb,
    RandomSoxReverb,
    RandomPedalboardDelay,
    RandomPedalboardCompressor,
    RandomPedalboardDistortion,
    RandomPedalboardLimiter,
    RandomPedalboardChorus,
    RandomPedalboardPhaser,
    RandomParametricEQ,
    RandomStereoWidener,
    RandomVolumeAutomation,
    LoudnessNormalize,
)


if __name__ == "__main__":

    sample_rate = 44100.0

    transforms = [
        RandomApply([RandomParametricEQ(sample_rate)], p=0.7),
        RandomApply([RandomPedalboardDistortion(sample_rate)], p=0.01),
        RandomApply([RandomPedalboardDelay(sample_rate)], p=0.1),
        RandomApply([RandomPedalboardChorus(sample_rate)], p=0.01),
        RandomApply([RandomPedalboardPhaser(sample_rate)], p=0.01),
        RandomApply([RandomPedalboardCompressor(sample_rate)], p=0.5),
        RandomApply([RandomPebalboardReverb(sample_rate)], p=0.2),
        RandomApply([RandomStereoWidener(sample_rate)], p=0.5),
        RandomApply([RandomPedalboardLimiter(sample_rate)], p=0.1),
        RandomApply([RandomVolumeAutomation(sample_rate)], p=0.7),
        LoudnessNormalize(sample_rate, target_lufs_db=-32.0),
    ]

    transforms = Compose(transforms)

    # load stems from a song
    rootdir = "/import/c4dm-datasets-ext/musdb18hq/train/James May - Dont Let Go"
    stems = []
    for stem_name in ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]:
        x, sr = torchaudio.load(os.path.join(rootdir, stem_name))
        stems.append(x[:, int(44100 * 12) : int(44100 * 22)])

    stems = torch.stack(stems, dim=0)
    print(stems.shape)

    for n in range(25):
        mix, processed_stems = transforms((None, stems))

        for stem_idx in range(processed_stems.shape[0]):
            torchaudio.save(
                f"outputs/{n}-{stem_idx}.wav", processed_stems[stem_idx, ...], sr
            )

        torchaudio.save(f"outputs/{n}-mix.wav", mix.view(2, -1), sr)
