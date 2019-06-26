################################################### This code takes the text and the ccorrespomding speech, batching, data loading and shuffling #####################################

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




######## extract spectrogram and pad it with zeros #######

def load_wav(filename):
  print("filname is", filename)
#  fs_slt,x_slt = wavfile.read(filename)
  data = librosa.load(filename, sr=16000)
  print("laod wav is", data[0])
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


###### Data Loading Class ######


class arctic_database(Dataset):

   def __init__(self, src):

    self.src = src

   def __len__(self):

    print("len is", len(self.src))
    return len(self.src)

   def __getitem__(self, idx):

    return self.src[idx]


###############################


##### Padding the sequences ######

def pad_seq(sequence):

  ordered = sorted(sequence, key=len, reverse=True)
  lengths = [len(x) for x in ordered]
  max_length = lengths[0]
#  print("max len is", max_length)
  seq_len = [len(seq) for seq in sequence]
#  print("seq len is", seq_len)
 # print("seq len is:", seq_len)
  padded = []
  for i in range(0,len(sequence)):
   npad = ((0, max_length-len(sequence[i])),(0,0))
#   print("n pad shape is", numpy.shape(npad), npad)
   padded.append(np.pad(sequence[i], pad_width=npad, mode='constant', constant_values = 0))
  return padded, lengths


##################################


####### Sorting the Sequences ########

def sort_batch(batch, lengths):
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    return seq_tensor,  seq_lengths

######################################


####### Loading the inputs and output files ####

Train_input=[]

input_folder = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/toy_data/wav'
os.chdir(input_folder)
input_files =sorted(os.listdir(input_folder))
for file in input_files:

   file_read = (load_wav(file))### numpy.loadtxt
   Train_input.append(file_read)
#print("train inp is", len(Train_input))

Train_input = np.array(Train_input)
#print("after array thing", len(Train_input))
padded_input, lengths = pad_seq(Train_input)
#print("shape os padded input", numpy.shape(padded_input))
#inp,lens = sort_batch(torch.Tensor(padded_input), lengths)


eng_data = arctic_database(padded_input)

datass = DataLoader(eng_data, batch_size=4, shuffle=True)

################################################


######## Calling the DataLoader ########


for slt in datass:

    print("slt is", numpy.shape(slt.data), numpy.shape(slt.data.transpose(1,2)))


'''

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
     #print("spec is", [spectrogram(w) for w in wave])
#     print("before", [numpy.shape(wa) for wa in wave])
#     print("after", numpy.shape(_prepare_data(wave)))
     wave = _prepare_data(wave)
#     print("prapre data is", wave)
     magnitude = np.array([spectrogram(w) for w in wave])
     print("magnitude is", magnitude)

'''
