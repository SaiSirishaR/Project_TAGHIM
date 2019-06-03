import os
import numpy as np
from symbols import symbols
import numpy

folder = '/home2/srallaba/projects/stft_feats'
os.chdir(folder)
files = sorted(os.listdir(folder))
for file in files:
   f = np.loadtxt(file)

   print(" f is", numpy.shape(f))



'''
#print ("symbols are:",  [s for s in symbols])
_symbol_to_id = {s: i for i, s in enumerate(symbols)}



def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
 return s in _symbol_to_id and s is not '_' and s is not '~'


def text_to_seq(text):
 sequence = []

 #sequence += _symbols_to_sequence(_clean_text(text))
 sequence +=_symbols_to_sequence(text)
 print("seq is", sequence)

file = open('/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/txt.done.data')
p = file.read()
print("p is", p)
text_to_seq(p)

import librosa
import numpy as np
import os

folder = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/orig_wav'
new_folder = '/home2/srallaba/projects/stft_feats'
os.chdir(folder)
files = sorted(os.listdir(folder))
for file in files:

 print("file is", file)
 fs_slt,x_slt = wavfile.read(file)

'''
