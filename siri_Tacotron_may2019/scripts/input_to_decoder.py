# This code is to extract speech features (spectrogram) and feed them to decoder

import os
import librosa
import numpy
import numpy as np
from scipy import signal

######## Parameters for feature extraction ########

num_freq = 1024
sample_rate = 16000
frame_length_ms = 50.
frame_shift_ms = 12.5
preemphasis = 0.97
ref_level_db = 20
min_level_db = -100
power = 1.5
griffin_lim_iters = 60
n_fft = (num_freq - 1) * 2
hop_length = int(frame_shift_ms / 1000 * sample_rate)
win_length = int(frame_length_ms / 1000 * sample_rate)

specgram_array = []

###### Loading wavefiles, preprocessing, spectrogram extraction ######

def load_wav(filename):

   wav_data = librosa.load(filename, sr=16000)
   return wav_data[0]

def preemphasis(file):
 print("entering pre-emphasis")
 return signal.lfilter([1, -0.97], [1], file)

def stft(file):
  return librosa.stft(y=file, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _amp_to_db(x):
 return 20 * np.log10(np.maximum(1e-5, x))

def _normalize(S):
 return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def spectrogram(file):
   print("entering spectrogram function")
   preprocessing = stft(preemphasis(file)) # stft over pre-emphasized signal
   print("prephasized", preprocessing)
   S = _amp_to_db(np.abs(preprocessing)) - ref_level_db # further processing over stft feats
   print ("feats before normalisation", S)
   return _normalize(S) # returning normalised speech feats



######## Modules for speech reconstruction ########

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def preemphasis(x):
    return signal.lfilter([1, -0.97], [1], x)


def inv_preemphasis(x):
    return signal.lfilter([1], [1, -0.97], x)

def inv_spectrogram(spectrogram):

    S = _denormalize(spectrogram)
    print("denormalised")
    S = _db_to_amp(S + ref_level_db)  # Convert back to linear

    return inv_preemphasis(_griffin_lim(S ** power))          # Reconstruct phase

def _griffin_lim(S):

    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(stft(y)))
        y = _istft(S_complex * angles)
    return y

def _istft(y):
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)

def _denormalize(S):
 return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)

####### Importing data for abs test ########

folder = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/toy_data/wav' # folder containing the wavfiles
os.chdir(folder) #entering the directory with the comand change directory
files = sorted (sorted(os.listdir(folder))) # sort them according to the order they are stored.
for file in files:
  specgram = spectrogram(load_wav(file)) # load wavefile and extract spectrogram do not use same name on the LHS (spectrogram) gives error numpyndarray cannot be called
