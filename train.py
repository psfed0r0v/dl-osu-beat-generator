import torch
from torch import nn
import wandb
import datetime
from config import params


def train(model, trainloader):
    if not params.ITER_LOG:
        ITER_LOG = trainloader.__len__() - 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    now = datetime.datetime.now()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(params.N_EPOCHS):
        loss_log = 0.0
        for i, data in enumerate(trainloader, 0):
            mels, labels = data[0].to(device), data[1].to(device)

            pred = model(mels.unsqueeze(-1).permute(0, 3, 1, 2))
            optimizer.zero_grad()
            loss = criterion(pred.float(), labels.float())
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

    return model
