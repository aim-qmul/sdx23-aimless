from encodec import EncodecModel
from encodec.utils import convert_audio

import os
import argparse
import torchaudio
import torch
import glob
from tqdm import tqdm


def sdr(input, target):
    num = (target[0, :] ** 2).sum() + (target[1, :] ** 2).sum()
    den1 = ((target[0, :] - input[0, :]) ** 2).sum()
    den2 = ((target[1, :] - input[1, :]) ** 2).sum()
    value = 10 * torch.log10(num / (den1 + den2).clamp(1e-8))
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rootdir",
        help="Path to directory containing dataset to encode.",
    )
    parser.add_argument(
        "--output",
        help="Root directory to store encoded dataset.",
        default=None,
    )
    parser.add_argument(
        "--gpu",
        help="Run codec on GPU.",
        action="store_true",
    )
    args = parser.parse_args()

    if args.output is None:
        args.rootdir = args.rootdir.rstrip("/")
        input_basename = os.path.basename(args.rootdir)
        input_dirname = os.path.dirname(args.rootdir)
        output_dir = os.path.join(input_dirname, f"{input_basename}-encodec")
        print(output_dir)
        os.makedirs(output_dir)

    # Instantiate a pretrained EnCodec model
    model = EncodecModel.encodec_model_48khz()
    model.set_target_bandwidth(24.0)

    if args.gpu:
        model.cuda()

    # find all song directories
    for subset in ["train", "test"]:
        directories = glob.glob(os.path.join(args.rootdir, subset, "*"))
        directories = [
            directory for directory in directories if os.path.isdir(directory)
        ]
        print(f"Found {len(directories)} directories.")

        # encode each directory
        for directory in tqdm(directories):
            output_song_dir = os.path.join(
                output_dir, subset, os.path.basename(directory)
            )
            # make new directory to store these outputs
            os.makedirs(output_song_dir)
            print(output_song_dir)

            # Load and pre-process the audio waveforms
            for source in ["mixture", "bass", "vocals", "other", "drums"]:
                filepath = os.path.join(directory, f"{source}.wav")
                wav, sr = torchaudio.load(filepath)
                wav = convert_audio(wav, sr, model.sample_rate, model.channels)
                wav = wav.unsqueeze(0)
                wav = wav.cuda()

                # Extract discrete codes from EnCodec
                with torch.no_grad():
                    encoded_frames = model.encode(wav)
                # codes = torch.cat(
                #    [encoded[0] for encoded in encoded_frames], dim=-1
                # )  # [B, n_q, T]

                filepath = os.path.join(output_song_dir, f"{source}.pt")
                torch.save(encoded_frames, filepath)
                # print(filepath)

                # with torch.no_grad():
                #    decoded_frames = model.decode(encoded_frames)

                # trim excess from input
                # decoded_frames = decoded_frames[..., : wav.shape[-1]]

                # error = sdr(decoded_frames.view(2, -1), wav.view(2, -1))
                # print(error)
                # torchaudio.save(
                #    f"encodec-{source}.wav",
                #    decoded_frames.view(2, -1),
                #    model.sample_rate,
                # )
