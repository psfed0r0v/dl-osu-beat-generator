from utils.utils import DictX

params = DictX({
    'MODE': 'normal',
    'DATA_PATH': 'data/',
    'CUT_RATE_SEC': 5,
    'RANDOM_SEED': 42,

    # model params
    'OUT_SHAPE': 5001,
    'N_EPOCHS': 3,
    'LEARNING_RATE': 0.001,
    'ITER_LOG': 50,
    'BATCH_SIZE': 16,
    'MODEL_PATH': 'tempo_model.pth'
})
