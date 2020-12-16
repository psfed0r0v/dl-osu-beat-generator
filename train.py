import wandb
import torch
# import torchvision
from torch import nn
# !pip install torchaudio
# !pip install wandb
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import datetime


def train(model, N_EPOCHS, trainloader, lr=0.001, BATCH_SIZE=128, ITER_LOG=None):
    if not ITER_LOG:
        ITER_LOG = trainloader.__len__() - 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    now = datetime.datetime.now()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(N_EPOCHS):
        loss_log = 0.0
        for i, data in enumerate(trainloader, 0):
            mels, labels = data[0].to(device), data[1].to(device)
            pred = model(mels)

            optimizer.zero_grad()
            loss = criterion(pred, labels)
            loss_log += loss.item()
            loss.backward()
            optimizer.step()

            if i % ITER_LOG == ITER_LOG - 1:
                wandb.log({"loss": loss_log / ITER_LOG})

                print('[%d, %5d] Running loss: %.3f' %
                      (epoch + 1, i + 1, loss_log / ITER_LOG))
                loss_log = 0.0

                print('time:', datetime.datetime.now() - now)
                now = datetime.datetime.now()

                print(' ')

    print('Finished Training')
