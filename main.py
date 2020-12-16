from dataset import DatasetNorm
from torch.utils.data import DataLoader
import torch
import wandb

from utils.read_data import parse_data
from utils.utils import set_random_seed, save_model, load_model
from config import params
from model.model import TempoCNN
from train import train


def main():
    set_random_seed(params.RANDOM_SEED)
    parse_data()
    data = DatasetNorm('cutted_data')
    train_set, test_set = torch.utils.data.random_split(data, [data.__len__() - 100, 100])
    trainloader = DataLoader(dataset=train_set, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=8)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tcnn = TempoCNN().to(device)

    wandb.init(project="tcnn")
    config = wandb.config
    config.learning_rate = 0.001
    wandb.watch(tcnn)

    model = train(tcnn, trainloader)
    save_model(model)


if __name__ == '__main__':
    main()
