import numpy as np
import torch
import random
from model.model import TempoCNN
from config import params


class DictX(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'


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
