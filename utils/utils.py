import numpy as np
import torch
import random
from model.model import TempoCNN
from config import get_params

params = get_params()


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_model(model):
    torch.save(model.state_dict(), params.MODEL_PATH)


def load_model():
    model = TempoCNN()
    model.load_state_dict(torch.load(params.MODEl_PATH))
    model.eval()


def accuracy(preds, labels):
    accuracy = 0.0
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if preds[i][j] > 0.15 and labels[i][j] == 1:
                accuracy += 1

    return accuracy / torch.sum(labels)
