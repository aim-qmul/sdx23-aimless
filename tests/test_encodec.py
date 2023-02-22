import torch
from encodec import EncodecModel
from encodec.utils import convert_audio

from aimless.models.codeclm import CodecLM

if __name__ == "__main__":

    codec = EncodecModel.encodec_model_48khz()
    codec.set_target_bandwidth(24.0)

    sources = ["bass", "drums", "vocals", "other"]
    n_src = len(sources)
    codeclm = CodecLM(sources, codec.quantizer.n_q, codec.quantizer.bins)

    mixture = torch.randn(3, 2, 131072)
    sources = torch.randn(3, n_src, 2, 131072)

    mixture_codebooks = codec.encode(mixture)
    print(len(mixture_codebooks))

    mixture_codebooks = torch.cat([mc[0] for mc in mixture_codebooks], dim=-1)
    print(mixture_codebooks.shape)

    source_codes = []
    for sidx in range(n_src):
        source_codebook = codec.encode(sources[:, sidx, ...])
        source_codebook = torch.cat([mc[0] for mc in source_codebook], dim=-1)
        source_codes.append(source_codebook)

    source_codes = torch.stack(source_codes, dim=1)
    print(source_codes.shape)

    criterion = torch.nn.NLLLoss()

    print(mixture_codebooks.shape)
    out = codeclm(mixture_codebooks)
    print(out.shape)

    loss = criterion(out, source_codes)
    print(loss)

    # sample max from each prediction and decoder
    out = torch.argmax(out, dim=1)
    print(out.shape)

    for n in range(n_src):
        print(out[:, n, ...].shape)
        audio = codec.decode(out[:, n, ...])
        print(audio.shape)
