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
    model.load_state_dict(torch.load(params.MODEL_PATH))
    model.eval()
    return model


def accuracy(preds, labels, threshold=0.5):
    preds = torch.sigmoid(preds)
    accuracy = 0.0
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if preds[i][j] > threshold and labels[i][j] == 1:
                accuracy += 1

    return accuracy / torch.sum(labels)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc / len(y_pred_tag)
