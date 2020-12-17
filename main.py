# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# import wandb

# from config import get_params
# from dataset import DatasetNorm
from utils.read_data import parse_data
# from utils.utils import set_random_seed, save_model, load_model, accuracy
# from model.model import TempoCNN
# from train import train


# def main():
#     params = get_params()
#     set_random_seed(params.RANDOM_SEED)
#     # parse_data()
#     data = DatasetNorm('cutted_data')
#     train_set, test_set = torch.utils.data.random_split(data, [data.__len__() - 100, 100])
#     trainloader = DataLoader(dataset=train_set, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=8)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     tcnn = TempoCNN().to(device)
#
#     # wandb.init(project="tcnn")
#     # config = wandb.config
#     # config.learning_rate = 0.001
#     # wandb.watch(tcnn)
#
#     if not params.LOAD_MODEL:
#         model = train(tcnn, trainloader)
#         save_model(model)
#     else:
#         model = load_model().to(device)
#
#     testloader = DataLoader(dataset=test_set, batch_size=params.BATCH_SIZE, shuffle=True)
#
#     iters = 0
#     loss = 0.0
#     cr_loss = nn.BCELoss()
#     for i, data in enumerate(testloader, 0):
#         tcnn.eval()
#         mels, labels = data[0].to(device), data[1].to(device)
#         pred = model(mels.unsqueeze(-1).permute(0, 3, 1, 2)).to('cpu').detach()
#         res = accuracy(pred, labels)
#         print(res)
#
#         loss += cr_loss(pred.float(), labels.float().to('cpu').detach()).item()
#         iters += 1
#
#     print(loss / iters)


if __name__ == '__main__':
    parse_data()
