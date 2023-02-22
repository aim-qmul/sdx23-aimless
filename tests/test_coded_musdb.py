import torch

from data.coded_musdb import CodedMUSDB


dataset = CodedMUSDB("/import/c4dm-datasets-ext/sdx-2023/MUSDB18-7-WAV-encodec/train")

for idx, example in enumerate(dataset):
    x, y = example
    print(idx, x.shape, y.shape)
