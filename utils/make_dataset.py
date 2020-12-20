import zipfile
import os
import torch
import torchaudio

def make_dataset(dir, audio_dir, text_dir, enum_from=1):
  tracks = [dir + '/'+track for track in os.listdir(dir)]

  counter = enum_from
  for track in tracks:

    with zipfile.ZipFile(track, 'r') as zip_ref:
        files = zip_ref.namelist()
        found_audio = 0
        found_txt = 0
        audio_file = ''
        text_file = ''
        for f in files:
          if f == 'audio.mp3':
            found_audio+=1
            audio_file = f
          
          if '.osu' in f:
            found_txt+=1
            text_file=f

          if found_txt == 1 and found_audio ==1:
            found_audio = 0
            found_txt = 0

            zip_ref.extract(audio_file ,audio_dir)
            os.rename((audio_dir + '/'+ audio_file), (audio_dir+ '/'+str(counter)+'.mp3'))
            
            zip_ref.extract(text_file,text_dir)
            os.rename((text_dir + '/'+ text_file), (text_dir+ '/'+str(counter)+'.osu'))

            counter+=1
            break
            
            
def lower_quality(audio_dir, new_sr):
  tracks = [audio_dir + track for track in os.listdir(audio_dir)]
  for track in tracks:
    wav, sr = torchaudio.load(track)
    wav = torchaudio.transforms.Resample(sr, new_sr)(wav)
    torchaudio.save(track, wav , new_sr)
