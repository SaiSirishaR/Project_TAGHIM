################################################### Modified code for Data loading. This code is till spectrogram extraction  #####################################

import os
import scipy.io.wavfile as wavfile
from torch.utils.data import Dataset, DataLoader
import numpy as np
from symbols import symbols
import librosa
from scipy import signal
import numpy

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

n_fft = (1024 - 1) * 2
hop_length = int(12.5 / 1000 * 16000)
win_length = int(50 / 1000 * 16000)
min_level_db = -100

def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
 return s in _symbol_to_id and s is not '_' and s is not '~'


def text_to_seq(text):
 sequence = []

 sequence +=_symbols_to_sequence(text)


 return sequence


def _pad_data(x, length):

    _pad = 0
    return np.pad(x, (0, length - len(x)), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])



######## extract spectrogram and pad it with zeros #######

def load_wav(filename):
#  print("filname is", filename)
#  fs_slt,x_slt = wavfile.read(filename)
  data = librosa.load(filename, sr=16000)
#  print("laod wav is", data[0])
  return data[0]

#def _stft_parameters():
# return n_fft, hop_length, win_length


def _amp_to_db(x):
 return 20 * np.log10(np.maximum(1e-5, x))

def preemphasis(x):
 return signal.lfilter([1, -0.97], [1], x)

def _normalize(S):
 return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def spectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - 20
    return _normalize(S)


def _stft(y):
 #   n_fft, hop_length, win_length = _stft_parameters()
 return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

##############################################################

###### Class I am not loading the wavefiles here, just shuffling and taking the indexes ######


class arctic_database(Dataset):
    
    def __init__(self, txt_file, wav_dir):
        self.txt_file = txt_file
        self.wav_dir= wav_dir
#        print("entering class", self.wav_name)

    def __len__(self):
#        print("len of txt file is", len(self.txt_file))
        return len(self.txt_file)

    def __getitem__(self, idx):
 
        os.chdir(self.wav_dir)
        files = sorted(os.listdir(self.wav_dir))
        print("idx is", idx, "sending", files[idx])
        return files[idx]

####### Loading the inputs and output files ####


text_array = []
text_file = open('/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/toy_data/txt.done.data')
for line in text_file:
  text_array.append(line.split('\n')[0])

wav_name=[]

folder = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/toy_data/wav'
print("fiile going is", folder)
datas = arctic_database(text_array, folder)

train_loader = DataLoader(dataset = datas, batch_size=4, shuffle=True)


################################################


######## Calling the DataLoader ########


for slt in train_loader:

    print("slt is", slt)
    print("am loading wavefiles", [load_wav(p) for p in slt])
    wave =  [load_wav(p) for p in slt]
    wave = _prepare_data(wave)
    print("prapre data is", wave)
    magnitude = np.array([spectrogram(w) for w in wave])
    print("magnitude is", magnitude)

