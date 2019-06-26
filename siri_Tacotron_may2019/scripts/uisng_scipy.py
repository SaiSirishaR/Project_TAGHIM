import os
import scipy.io.wavfile as wavfile
import numpy
import numpy as np
import librosa

def load_wav(filename):
  print("filename is", filename)
  fs_slt,x_slt = wavfile.read(filename)
  data = librosa.load(filename, sr=16000)
 
  print("laod wav is", numpy.shape(x_slt), x_slt, len(data[0]))
  return x_slt


wav_name=[]

folder = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/toy_data/wav'
os.chdir(folder)
files = sorted(os.listdir(folder))
for file in files:

  wav_data = np.asarray(load_wav(file), dtype=np.float32)
  print("wav_data is", wav_data[0])
