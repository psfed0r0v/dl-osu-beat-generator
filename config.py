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


def get_params():
    return DictX({
        'MODE': 'normal',
        'DATA_PATH': 'data/',
        'CUT_RATE_SEC': 5,
        'RANDOM_SEED': 42,

        # model params
        'OUT_SHAPE': 5000,
        'N_EPOCHS': 5,
        'LEARNING_RATE': 0.0003,
        'ITER_LOG': 50,
        'BATCH_SIZE': 32,
        'MODEL_PATH': 'tempo_model.pth',
        'LOAD_MODEL': False
    })
