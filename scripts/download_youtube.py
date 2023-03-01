import torch
import glob
import torchaudio
from torchaudio.utils import download_asset
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade
from tqdm import tqdm
from audio_data_pytorch import YoutubeDataset, Resample
import os

bundle = HDEMUCS_HIGH_MUSDB_PLUS
sample_rate = bundle.sample_rate
fade_overlap = 0.1
subset = "train"
root = "/import/c4dm-datasets-ext/sdx-2023/youtube_dataset"
print(f"Sample rate: {sample_rate}")
ext = "flac"


def main():
    url_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"{subset}_urls.txt"
    )

    with open(url_file_path) as f:
        urls = f.readlines()

    print("Loading dataset...")
    dataset = YoutubeDataset(
        root=root,
        urls=urls,
        crop_length=60,  # Crop source in 10s chunks (optional but suggested)
        transforms=torch.nn.Sequential(
            Resample(source=48000, target=sample_rate),
            Fade(
                fade_in_len=0,
                fade_out_len=int(fade_overlap * sample_rate),
                fade_shape="linear",
            ),
        ),
    )
    print("Loading model...")
    model = bundle.get_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # locate all video directories
    video_dirs = glob.glob(os.path.join(root, "youtube_dataset", "*"))
    video_dirs = [vd for vd in video_dirs if os.path.isdir(vd)]

    batch_size = 2

    print("Separating chunks...")
    for i, video_dir in enumerate(video_dirs):
        video_uri = os.path.basename(video_dir)
        print(f"{i}/{len(video_dirs)} - {video_uri}")
        # find all chunks in this video dir
        chunk_filepaths = glob.glob(os.path.join(video_dir, "*.wav"))

        chunks = []
        chunk_idxs = []
        for chunk_filepath in tqdm(chunk_filepaths):
            chunk_idx = os.path.basename(chunk_filepath).replace(".wav", "")
            chunk, sr = torchaudio.load(chunk_filepath)
            chunk = torchaudio.functional.resample(chunk, sr, sample_rate)

            chunks.append(chunk)
            chunk_idxs.append(chunk_idx)

            # separate and reset buffer
            if len(chunks) >= batch_size:
                chunks = torch.stack(chunks, dim=0)
                sources = separate(chunks, model=model, device=device)

                for idx, chunk_idx in enumerate(chunk_idxs):
                    if not os.path.exists(f"{root}/{subset}/{video_uri}_{chunk_idx}"):
                        os.makedirs(f"{root}/{subset}/{video_uri}_{chunk_idx}")

                    torchaudio.save(
                        f"{root}/{subset}/{video_uri}_{chunk_idx}/mixture.flac",
                        chunks[idx, ...],
                        sample_rate=sample_rate,
                    )

                    for source in sources:
                        torchaudio.save(
                            f"{root}/{subset}/{video_uri}_{chunk_idx}/{source}.flac",
                            sources[source][idx, ...],
                            sample_rate=sample_rate,
                        )

                # reset buffers
                chunks = []
                chunk_idxs = []


def separate(chunk: torch.Tensor, model: torch.nn.Module, device: torch.device):
    chunk = chunk.to(device)
    ref = chunk.mean(0)
    chunk = (chunk - ref.mean()) / ref.std()  # normalization

    with torch.no_grad():
        out = model.forward(chunk)

    sources = out.squeeze(0)
    sources = sources * ref.std() + ref.mean()  # denormalization
    sources = sources.cpu()

    sources_list = model.sources
    dict_sources = {}
    for idx, source_name in enumerate(sources_list):
        dict_sources[source_name] = sources[:, idx, ...]

    return dict_sources


if __name__ == "__main__":
    main()
