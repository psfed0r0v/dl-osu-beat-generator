
from utils.read_data import parse_data
from utils import make_dataset

if __name__ == '__main__':
    make_dataset(params.OSU_TRACKS_DIR, params.DATA_PATH + '/audio_normal', + params.DATA_PATH + '/text_normal', params.ENUMERATE_FROM)
    parse_data()