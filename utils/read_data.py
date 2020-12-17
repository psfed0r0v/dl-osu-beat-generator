from config import get_params
import os
from os.path import isfile, join
from pathlib import Path
from pydub import AudioSegment
import math

params = get_params()


class SplitWavAudio:
    def __init__(self, filename, osu_filename, folder, output_folder):
        self.folder = folder
        self.output_folder = output_folder
        self.filename = filename
        self.filepath = folder + filename
        self.osu_filename = osu_filename

        self.audio = AudioSegment.from_wav(self.filepath + '.wav')
        self.points = None

    def get_duration(self):
        return self.audio.duration_seconds

    def write_file(self, t1, t2, file_name):
        res = []
        for dot in self.points:
            if dot >= t1 and dot <= t2:
                res.append(dot - t1)

        with open(file_name, 'w') as f:
            if len(res):
                for dot in res:
                    f.write(str(dot) + '\n')
            else:
                f.write('')

    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 1000
        t2 = to_min * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.output_folder + f'audio_{params.MODE}/' + split_filename + '.wav', format="wav")
        self.write_file(t1, t2, self.output_folder + f'text_{params.MODE}/' + split_filename + '.txt')

    def multiple_split(self, sec_per_split):
        with open(self.folder + self.osu_filename, 'r') as f:
            out = f.readlines()
        out_norm = []
        slidermultiplier = 0
        timings = {}
        timing_flag = False
        last_inherited = 0
        for i, s in enumerate(out):
            # if s != '\n':
            if 'SliderMultiplier' in s:
                slidermultiplier = float(s.split(':')[1])
            if timing_flag and (s == '' or s == '\n'):
                timing_flag = False
            if timing_flag:
                splits = s.split(',')
                if splits[6] == '1':
                    timings[int(splits[0].split('.')[0])] = (int(splits[1].split('.')[0]), 1)
                    last_inherited = int(splits[1].split('.')[0])
                else:
                    timings[int(splits[0].split('.')[0])] = (last_inherited, abs(100 / float(splits[1].split('.')[0])))
            if 'TimingPoints' in s:
                timing_flag = True
            if 'HitObjects' in s:
                out_norm = out[i + 1:]
                break
        res = []
        for i in out_norm:
            splits = i.split(',')
            res.append(int(splits[2]))
            if int(splits[3]) % 16 - 4 == 2 or int(splits[3]) % 16 == 2:
                times = list(timings.keys())
                timing = -1
                for j in range(len(times) - 1):
                    if int(splits[2]) < times[j + 1] and int(splits[2]) >= times[j]:
                        timing = times[j]
                        break
                if timing == -1:
                    timing = times[len(times) - 1]
                px_per_beat = slidermultiplier * 100 * timings[timing][1]
                beats_number = float(splits[7]) * int(splits[6]) / px_per_beat
                duration = math.ceil(beats_number * timings[timing][0])
                endtime = int(splits[2]) + duration
                res.append(endtime)
        self.points = res

        total_sec = int(self.get_duration())
        for i in range(0, total_sec - total_sec % 5, sec_per_split):
            split_fn = self.filename + '_' + str(i)
            self.single_split(i, i + sec_per_split, split_fn)


def parse_data():
    cur = os.getcwd()
    Path(cur + '/cutted_data').mkdir(parents=True, exist_ok=True)
    Path(cur + f'/cutted_data/audio_{params.MODE}').mkdir(parents=True, exist_ok=True)
    Path(cur + f'/cutted_data/text_{params.MODE}').mkdir(parents=True, exist_ok=True)

    audios = [f for f in os.listdir(params.DATA_PATH + 'audio_' + params.MODE) if
              isfile(join(params.DATA_PATH + 'audio_' + params.MODE, f))]
    for i in range(1, len(audios)):
        sound = AudioSegment.from_mp3(params.DATA_PATH + f'audio_{params.MODE}/' + str(i) + '.mp3')
        # print(i, mediainfo(params.DATA_PATH + f'audio_{params.MODE}/' + str(i) + '.mp3'))
        sound.export(params.DATA_PATH + f'audio_{str(i)}.wav', format='wav')
        tmp = SplitWavAudio(f'audio_{str(i)}', f'text_{params.MODE}/' + str(i) + '.osu', params.DATA_PATH,
                            'cutted_data/')
        tmp.multiple_split(params.CUT_RATE_SEC)
        path = Path(params.DATA_PATH + f'audio_{str(i)}.wav')
        path.unlink()

    print('data parsed')
