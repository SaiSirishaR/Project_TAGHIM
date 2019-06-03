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
        self.txt_file = txt_file
#        print("text is...:", text_to_seq(self.txt_file))
        self.wav_dir= wav_dir
        print("wav dir is", self.wav_dir)
        self.wav_name = wav_name

    def load_wav(self, filename):
        print("loading wavefile")
        return librosa.load(filename, sr=16000)

    def __len__(self):
        print("len is:", len(self.txt_file))
        return len(self.wav_dir)

    def __getitem__(self, idx):
       
#       return {'text':  self.txt_file[idx], 'wav': self.wav_dir[idx]}
       return self.txt_file[idx], self.wav_dir[idx]

text_array = []
text_file = open('/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/txt.done.data')
for line in text_file:
  text_array.append(line.split('\n')[0])

wav_name=[]

folder = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/orig_wav'
os.chdir(folder)
files = sorted(os.listdir(folder))
for file in files:
  wav_name.append(file)
  #print("wav name is", wav_name[0])
datas = CMUarctic(text_array, files, wav_name)
train_loader = DataLoader(dataset = datas, batch_size=4, shuffle=True)

for txti, wavi in train_loader:
 
     print("text is", txti, "wav is:", wavi)
