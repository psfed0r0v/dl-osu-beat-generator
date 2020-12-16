import wandb
import torch 
# import torchvision
from torch import nn
# !pip install torchaudio
# !pip install wandb
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def Norm(mel, mean, std, eps=1e-8):
  mel = torch.log(mel + eps)
  mel = (mel - mean) / std
  return mel

class dataset(Dataset):
    def __init__(self, wav_dir, n_mels, transform=None):
        data = [d for d in os.listdir(wav_dir)]
        self.data = data
        self.transform = transform
        self.n_mels = n_mels

    def __len__(self):
        return len(self.data)
    
    def noise(self,wav):
      wav += 0.01 * torch.randn(wav.shape)
      return wav

    def __getitem__(self, index):
        path = self.data[index]
        wav, sr = torchaudio.load(path + '/audio.wav')
        wav = torch.mean(wav,dim=0)
        points = pd.read_csv(path + '/audio.txt', sep=" ")
        target = np.zeros(5000)
        target[point] = 1.0 for point in points

        if self.transform:
            wav = self.transform(wav)
            wav = self.noise(wav)

        featurizer = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels)
        mels = featurizer(wav)
        mels = Norm(mels,torch.mean(mels),torch.std(mels))
        
        return mels, torch.tensor(target)