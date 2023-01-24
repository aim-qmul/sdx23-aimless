import torch
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
root = "./youtube_data"
print(f"Sample rate: {sample_rate}")


def main():
    url_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urls.txt")
    with open(url_file_path) as f:
        urls = f.readlines()
    print("Loading dataset...")
    dataset = YoutubeDataset(
        root=root,
        urls=urls,
        crop_length=10,  # Crop source in 10s chunks (optional but suggested)
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

    print("Separating chunks...")
    for i, chunk in enumerate(tqdm(dataset)):
        original_url_idx = int(dataset.wavs[i].split("/")[-1].split(".")[0])
        sources = separate(chunk, model=model, device=device)

        if not os.path.exists(f"{root}/separated/{original_url_idx}"):
            os.makedirs(f"{root}/separated/{original_url_idx}")
        for source in sources:
            torchaudio.save(
                f"{root}/separated/{original_url_idx}/{source}.wav",
                sources[source],  # Transpose to get channels first for soundfile
                sample_rate=sample_rate,
            )


def separate(chunk: torch.Tensor, model: torch.nn.Module, device: torch.device):
    chunk.to(device)
    ref = chunk.mean(0)
    chunk = (chunk - ref.mean()) / ref.std()  # normalization

    with torch.no_grad():
        out = model.forward(chunk[None])

    sources = out.squeeze(0)
    sources = sources * ref.std() + ref.mean()  # denormalization

    sources_list = model.sources
    sources = list(sources)
    dict_sources = dict(zip(sources_list, sources))
    return dict_sources


if __name__ == "__main__":
    main()
