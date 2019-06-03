import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from symbols import symbols
import librosa

_symbol_to_id = {s: i for i, s in enumerate(symbols)}



def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
 return s in _symbol_to_id and s is not '_' and s is not '~'


def text_to_seq(text):
 sequence = []

 sequence +=_symbols_to_sequence(text)


 return sequence

class CMUarctic(Dataset):

    def __init__(self, txt_file, wav_dir, wav_name):
        print("am in the function")
        self.txt_file = txt_file.read()
#        print("text is...:", text_to_seq(self.txt_file))
        self.wav_dir= wav_dir
        self.wav_name = wav_name

    def load_wav(self, filename):
        print("loading wavefile")
        return librosa.load(filename, sr=16000)

    def __len__(self):
        print("len is:", len(self.txt_file))
        return len(self.txt_file)

    def __getitem__(self, idx):
         return self.wav_dir[idx]
#        print("wav file is:", self.wav_name)
#        name = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/orig_wav'+ '/'+ self.wav_name
#        print("wav name is:", name)
#        text = self.landmarks_frame.ix[idx, 1]
#        print(" am in get item")
        text = text_to_seq(self.txt_file)
#        print("got text")
#        text = np.asarray(text_to_seq(self.txt_file), dtype=np.int32)
        wav = self.load_wav('/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/orig_wav'+ '/'+ self.wav_name)
#        wav = np.asarray(self.load_wav(files), dtype=np.float32)
        print("loaded wavfile", self.wav_name)
        sample = {'text': text, 'wav': wav}
#        print("sample is:", sample)
        return sample


text_file = open('/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/txt.done.data')
wav_name=[]

folder = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/orig_wav'
os.chdir(folder)
files = sorted(os.listdir(folder))
for file in files:
  wav_name.append(file)
  #print("wav name is", wav_name[0])
datas = CMUarctic(text_file, files, wav_name)
train_loader = DataLoader(dataset = datas, batch_size=4, shuffle=True)
for i in train_loader:
   print(i)
