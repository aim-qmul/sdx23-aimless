import argparse
import pathlib
import soundfile as sf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the MUSDB18 dataset")
    parser.add_argument("output", type=str, help="Path to the output folder")
    args = parser.parse_args()
    print("Converting MUSDB18 to voiceless")
    print("This will take a while ...")
    print("")

    path = pathlib.Path(args.path)
    output = pathlib.Path(args.output)
    output.mkdir(exist_ok=True)

    for track in path.glob("**/mixture.wav"):
        folder = track.parent
        print(f"Processing {folder}")

        bass, sr = sf.read(folder / "bass.wav")
        drums, sr = sf.read(folder / "drums.wav")
        other, sr = sf.read(folder / "other.wav")
        mix = bass + drums + other

        output_folder = output / folder.relative_to(path)
        output_folder.mkdir(exist_ok=True, parents=True)
        sf.write(output_folder / "mixture.wav", mix, sr)
