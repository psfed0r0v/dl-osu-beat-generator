import torch
from torch import nn
import wandb
import datetime
from config import get_params
from tqdm import tqdm
import matplotlib.pylab as plt
from utils.utils import accuracy, binary_acc
from sklearn.metrics import accuracy_score
import numpy as np


def train(model, trainloader):
    params = get_params()
    ITER_LOG = params.ITER_LOG

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    now = datetime.datetime.now()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.LEARNING_RATE)
    criterion = nn.BCELoss()
    losses = []
    for epoch in tqdm(range(params.N_EPOCHS)):
        loss_log = 0.0
        y_true = []
        y_pred = []
        for i, data in enumerate(trainloader, 0):
            mels, labels = data[0].to(device), data[1].to(device)

            pred = model(mels.unsqueeze(-1).permute(0, 3, 1, 2))
            optimizer.zero_grad()
            loss = criterion(pred.float(), labels.float())
            loss_log += loss.item()
            loss.backward()
            optimizer.step()
            pred = np.round(pred.to('cpu').detach())
            target = np.round(labels.to('cpu').detach())
            y_pred.extend(pred.tolist())
            y_true.extend(target.tolist())
            if i % ITER_LOG == ITER_LOG - 1:
                # wandb.log({"loss": loss_log / ITER_LOG})

                print('[%d, %5d] Running loss: %.3f' %
                      (epoch + 1, i + 1, loss_log / ITER_LOG))
                losses.append(loss_log / ITER_LOG)
                loss_log = 0.0

                print('time:', datetime.datetime.now() - now)
                now = datetime.datetime.now()

                # print('Acc:\t', accuracy(pred.to('cpu').detach(), labels.to('cpu').detach()).item())

                print('Acc:\t', accuracy_score(y_true, y_pred))

    plt.plot(list(range(params.N_EPOCHS)), losses)
    plt.show()

    print('Finished Training')

    return model
