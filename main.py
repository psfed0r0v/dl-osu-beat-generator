import os
from os.path import isfile, join
from pathlib import Path
from pydub import AudioSegment
from utils.read_data import SplitWavAudio
from config import params


def main():
    cur = os.getcwd()
    Path(cur + '/cutted_data').mkdir(parents=True, exist_ok=True)
    Path(cur + f'/cutted_data/audio_{params.MODE}').mkdir(parents=True, exist_ok=True)
    Path(cur + f'/cutted_data/text_{params.MODE}').mkdir(parents=True, exist_ok=True)

    audios = [f for f in os.listdir(params.DATA_PATH + 'audio_' + params.MODE) if
              isfile(join(params.DATA_PATH + 'audio_' + params.MODE, f))]
    for i in range(1, len(audios)):
        sound = AudioSegment.from_mp3(params.DATA_PATH + f'audio_{params.MODE}/' + str(i) + '.mp3')
        sound.export(params.DATA_PATH + f'audio_{str(i)}.wav', format='wav')
        tmp = SplitWavAudio(f'audio_{str(i)}', f'text_{params.MODE}/' + str(i) + '.osu', params.DATA_PATH, 'cutted_data/')
        tmp.multiple_split(params.CUT_RATE_SEC)
        path = Path(params.DATA_PATH + f'audio_{str(i)}.wav')
        path.unlink()

    print('OK')


if __name__ == '__main__':
    main()
