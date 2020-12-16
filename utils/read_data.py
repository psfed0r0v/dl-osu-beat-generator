from pydub import AudioSegment
from config import params

MODE = params['MODE']


class SplitWavAudio:
    def __init__(self, filename, osu_filename, folder, output_folder):
        self.folder = folder
        self.output_folder = output_folder
        self.filename = filename
        self.filepath = folder + filename
        self.osu_filename = osu_filename

        self.audio = AudioSegment.from_wav(self.filepath)
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
                f.write('\n')

    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 1000
        t2 = to_min * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.output_folder + f'audio_{MODE}/' + split_filename, format="wav")
        self.write_file(t1, t2, self.output_folder + f'text_{MODE}/' + split_filename + '.txt')

    def multiple_split(self, sec_per_split):
        with open(self.folder + self.osu_filename, 'r') as f:
            out = f.readlines()
        out_norm = []
        for i, s in enumerate(out):
            if 'HitObjects' in s:
                out_norm = out[i + 1:]
                break
        res = []
        for i in out_norm:
            res.append(int(i.split(',')[2]))
        self.points = res

        total_sec = int(self.get_duration())
        for i in range(0, total_sec, sec_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i + sec_per_split, split_fn)
