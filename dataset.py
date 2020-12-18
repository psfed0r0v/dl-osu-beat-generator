import torch
from torch.utils.data import Dataset
import torchaudio
import os
import pandas as pd
import numpy as np
from config import get_params

params = get_params()


def Norm(mel, mean, std, eps=1e-8):
    mel = torch.log(mel + eps)
    mel = (mel - mean) / std
    return mel


class DatasetNorm(Dataset):
    def __init__(self, wav_dir, n_mels=80, transform=None):
        data = [wav_dir + '/audio_normal/' + d for d in os.listdir(wav_dir + f'/audio_{params.MODE}')]
        self.data_audio = data
        data = [wav_dir + '/text_normal/' + d for d in os.listdir(wav_dir + f'/text_{params.MODE}')]
        self.data_txt = data
        self.wav_dir = wav_dir
        self.transform = transform
        self.n_mels = n_mels
        self.featurizer = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels, sample_rate=44100)

    def __len__(self):
        return len(self.data_audio)

    def noise(self, wav):
        wav += 0.01 * torch.randn(wav.shape)
        return wav

    def __getitem__(self, index):
        path_audio = self.data_audio[index]
        wav, sr = torchaudio.load(path_audio)
        wav = torch.mean(wav, dim=0)
        path_txt = self.data_txt[index]
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        target = np.zeros(params.OUT_SHAPE)
        delimeter = 5000 // (params.OUT_SHAPE - 1)
        if not os.stat(path_txt).st_size == 0:
            if not os.stat(path_txt).st_size == 0:
                points = pd.read_csv(path_txt, header=None).values[:, 0]
                for point in points:
                    p = (point - 1) // delimeter if delimeter != 1 else point - 1
                    target[p] = 1.0

        # if self.transform:
        #     wav = self.transform(wav)
        # wav = self.noise(wav)

        mels = self.featurizer(wav)
        # mels = Norm(mels, torch.mean(mels), torch.std(mels))

        return mels, torch.tensor(target)
