import os
from os.path import isfile, join
from pathlib import Path
from pydub import AudioSegment
from utils.read_data import SplitWavAudio
from config import params

DATA_PATH = params['DATA_PATH']
CUT_RATE_SEC = params['CUT_RATE_SEC']
MODE = params['MODE']


def main():
    cur = os.getcwd()
    Path(cur + '/cutted_data').mkdir(parents=True, exist_ok=True)
    Path(cur + f'/cutted_data/audio_{MODE}').mkdir(parents=True, exist_ok=True)
    Path(cur + f'/cutted_data/text_{MODE}').mkdir(parents=True, exist_ok=True)

    audios = [f for f in os.listdir(DATA_PATH + 'audio_' + MODE) if isfile(join(DATA_PATH + 'audio_' + MODE, f))]
    for i in range(1, len(audios)):
        sound = AudioSegment.from_mp3(DATA_PATH + f'audio_{MODE}/' + str(i) + '.mp3')
        sound.export(DATA_PATH + f'audio_{str(i)}.wav', format='wav')
        tmp = SplitWavAudio(f'audio_{str(i)}.wav', f'text_{MODE}/' + str(i) + '.osu', DATA_PATH, 'cutted_data/')
        tmp.multiple_split(CUT_RATE_SEC)
        path = Path(DATA_PATH + f'audio_{str(i)}.wav')
        path.unlink()

    print('OK')


if __name__ == '__main__':
    main()
