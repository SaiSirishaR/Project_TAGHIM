################################################### This code takes the text and the ccorrespomding speech, batching, data loading and shuffling #####################################

import os
import scipy.io.wavfile as wavfile
from torch.utils.data import Dataset, DataLoader
import numpy as np
from symbols import symbols
import librosa
from scipy import signal
import numpy
from network import *
import torch.nn as nn
import torch
from torch.autograd import Variable

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

n_fft = (1024 - 1) * 2
hop_length = int(12.5 / 1000 * 16000)
win_length = int(50 / 1000 * 16000)
min_level_db = -100
outputs_per_step = 5
griffin_lim_iters = 60

def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
 return s in _symbol_to_id and s is not '_' and s is not '~'


def text_to_seq(text):
 sequence = []

 sequence +=_symbols_to_sequence(text)


 return sequence


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(16000 * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)


###### Importing model #######

model = Tacotron()
######## extract spectrogram and pad it with zeros #######

def load_wav(filename):
  fs_slt,x_slt = wavfile.read(filename)
  return x_slt

#def _stft_parameters():
# return n_fft, hop_length, win_length


def _amp_to_db(x):
 return 20 * np.log10(np.maximum(1e-5, x))

def preemphasis(x):
 return signal.lfilter([1, -0.97], [1], x)

def _normalize(S):
 return np.clip((S - min_level_db) / -min_level_db, 0, 1)



def _denormalize(S):
 return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def _db_to_amp(x):
 return np.power(10.0, x * 0.05)

def inv_preemphasis(x):
 return signal.lfilter([1], [1, -0.97], x)

def spectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - 20
    return _normalize(S)

def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _istft(y):
#    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)

def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''

    S = _denormalize(spectrogram)
    S = _db_to_amp(S + 20)  # Convert back to linear

    return inv_preemphasis(_griffin_lim(S ** 1.5)) 

def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, outputs_per_step - (timesteps % outputs_per_step)]], mode='constant', constant_values=0.0)




def _stft(y):
 #   n_fft, hop_length, win_length = _stft_parameters()
 return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

##############################################################

class CMUarctic(Dataset):

    def __init__(self, txt_file, wav_dir, wav_name):
        self.txt_file = txt_file
        self.wav_dir= wav_dir
        self.wav_name = wav_name


    def __len__(self):
        return len(self.wav_dir)

    def __getitem__(self, idx):
       return self.txt_file[idx], self.wav_dir[idx]

def _pad_data(x, length):

    _pad = 0
    return np.pad(x, (0, length - len(x)), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


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
datas = CMUarctic(text_array, files, wav_name)
train_loader = DataLoader(dataset = datas, batch_size=4, shuffle=True)

for txti, wavi in train_loader:
 
     text = [text_to_seq(p) for p in txti]  

####### pad sequences #######  
    
     
     text = _prepare_data(text).astype(np.int32)
     wave = [load_wav(p) for p in wavi]
     wave = _prepare_data(wave)
#     print("prapre data is", wave)
     magnitude = np.array([spectrogram(w) for w in wave])
#     print("magnitude is", [numpy.shape(pp) for pp in magnitude], "shape is", magnitude.shape[-1], "before", numpy.shape(magnitude))
     magnitude = _pad_per_step(magnitude)
#     print("padded per time step", [numpy.shape(oo) for oo in magnitude], numpy.shape(magnitude))
     ##characters = Variable(torch.from_numpy(data[0]).type(torch.LongTensor), requires_grad=False)
     characters = Variable(torch.from_numpy(text).type(torch.LongTensor), requires_grad=False)
     magnitude = Variable(torch.from_numpy(magnitude).type(torch.FloatTensor), requires_grad=False)
     mag_output, linear_output = model.forward(characters, magnitude)

##### Reconstruction ########
     wav = inv_spectrogram(linear_output[0].data.cpu().numpy())
     wav = wav[:find_endpoint(wav)]
     wav *= 32767 / max(0.01, np.max(np.abs(wav)))
     librosa.output.write_wav('/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/scripts/resynth/file.wav', wav.astype(np.int16), 16000)
