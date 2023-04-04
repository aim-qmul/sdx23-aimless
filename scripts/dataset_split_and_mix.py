from pathlib import Path
from tqdm import tqdm
import torchaudio
import sys


# Run with python scripts/dataset_split_and_mix.py /path/to/dataset
def main():
    root = Path(sys.argv[1])
    splits = [0.9, 0.1, 0]

    train = root / "train"
    # train = Path("./") / "train"
    valid = root / "valid"
    # valid = Path("./") / "valid"
    # test = root / "test"
    train.mkdir(exist_ok=True)
    valid.mkdir(exist_ok=True)
    # test.mkdir(exist_ok=True)
    total_tracks = len(list(root.iterdir())) - 3  # 3 for train, valid, test
    num_train = int(total_tracks * splits[0])
    num_valid = int(total_tracks * splits[1])
    num_test = int(total_tracks * splits[2])

    # Add remainder as necessary
    remainder = total_tracks - (num_train + num_valid + num_test)
    for i in range(remainder):
        if i % 2 == 0:
            num_train += 1
        elif i % 2 == 1:
            num_valid += 1
        # else:
        #     num_test += 1

    num = 0
    for d in tqdm(root.iterdir(), total=total_tracks):
        if d.is_dir() and d.name != "train" and d.name != "valid" and d.name != "test":
            bass, sr = torchaudio.load(str(d / "bass.wav"))
            drums, sr = torchaudio.load(str(d / "drums.wav"))
            other, sr = torchaudio.load(str(d / "other.wav"))
            vocals, sr = torchaudio.load(str(d / "vocals.wav"))
            mixture = bass + drums + other + vocals
            torchaudio.save(str(d / "mixture.wav"), mixture, sr)

            if num < num_train:
                d.rename(train / d.name)
            # elif num < num_train + num_valid:
            else:
                d.rename(valid / d.name)
            # else:
            #     d.rename(test / d.name)
            num += 1


if __name__ == "__main__":
    main()
