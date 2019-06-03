##### This code extracts spectrogram from the wavfilee and resynthesizes speech waveform from the ectracted spectrogram #######

#!/usr/bin/python

import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import numpy as np
import os

folder = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/orig_wav'
new_folder = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/modified_wav'
os.chdir(folder)
files = sorted(os.listdir(folder))
for file in files:

 print("file is", file)
 fs_slt,x_slt = wavfile.read(file)

 x_slt = signal.lfilter([1, -0.97], [1], x_slt)

 min_level_db = -100

##### stft ########


 n_fft = (1024 - 1) * 2
 hop_length = int(12.5 / 1000 * 16000)
 win_length = int(50 / 1000 * 16000)

 stft = librosa.stft(y=x_slt, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
 
 def amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

 def normalize(S):
  return np.clip((S - min_level_db) / -min_level_db, 0, 1)


 def _denormalize(S):
  return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

 def _db_to_amp(x):
  return np.power(10.0, x * 0.05)


 def inv_preemphasis(x):
  return signal.lfilter([1], [1, -0.97], x)

 istft =  librosa.istft(stft, hop_length=hop_length, win_length=win_length)


 def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(16000 * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)


#### print("got stft", stft,"abs is",  np.abs(stft), " amp to db is", amp_to_db(np.abs(stft)), "unnormalise_value", amp_to_db(np.abs(stft))-20, "normalized", normalize(amp_to_db(np.abs(stft))-20))



# Spectrogram to wav
#_, linear_output = model.forward(characters, mel_input)
#wav = inv_spectrogram(linear_output[0].data.cpu().numpy())
 wav = istft
 wav = wav[:find_endpoint(wav)]

 def save_wav(wav):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
 librosa.output.write_wav(new_folder+'/'+file, wav.astype(np.int16), 16000)



#out = io.BytesIO()
 save_wav(wav)


 magnitude = np.array([p for p in  normalize(amp_to_db(np.abs(stft))-20)])
 #####print("magnitude is", magnitude, "shape is",  magnitude.shape[-1])
#plt.show()


