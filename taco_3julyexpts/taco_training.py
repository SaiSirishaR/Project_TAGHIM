import os
import numpy
from torch.utils.data import Dataset, DataLoader
import numpy as np
from symbols import symbols
import librosa
from scipy import signal
import numpy
import torch
from torch import *
from torch.autograd import Variable
import time
from network import *
from torch import optim

log_step = 100
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
n_fft = (1024 - 1) * 2
hop_length = int(12.5 / 1000 * 16000)
win_length = int(50 / 1000 * 16000)
min_level_db = -100
num_freq  = 1024
outputs_per_step = 5
batch_size =32

####### Modules for text processing #######


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
 return s in _symbol_to_id and s is not '_' and s is not '~'


def text_to_seq(text):

 sequence = []
 sequence +=_symbols_to_sequence(text)
 return sequence


############################################

###### Modules for Speech #####################

def load_wav(filename):

  data = librosa.load(filename, sr=16000)
  return data[0]

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
 return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


_mel_basis = None

def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
    n_fft = (num_freq - 1) * 2
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels)

def melspectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return _normalize(S)

def _pad_data(x, length):

    _pad = 0
    return np.pad(x, (0, length - len(x)), mode='constant', constant_values=_pad)

def _istft(y):
#    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(16000 * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)

def _griffin_lim(S):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y

def _denormalize(S):
 return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def _db_to_amp(x):
 return np.power(10.0, x * 0.05)

def inv_preemphasis(x):
 return signal.lfilter([1], [1, -0.97], x)


def inv_spectrogram(spectrogram):

    S = _denormalize(spectrogram)
    S = _db_to_amp(S + 20)  # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** 1.5))




#######################################################################


######### Modules common for both Text and Speech ########

def _prepare_data(inputs):

    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


def _pad_per_step(inputs):

    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, outputs_per_step - (timesteps % outputs_per_step)]], mode='constant', constant_values=0.0)


#########################################################

class train_database(Dataset):

    def __init__(self, txt_file, wav_dir):
        self.txt_file = txt_file
        self.wav_dir= wav_dir

    def __len__(self):
        return len(self.txt_file)

    def __getitem__(self, idx):

        os.chdir(self.wav_dir)
        files = sorted(os.listdir(self.wav_dir))
        return self.txt_file[idx], files[idx]

####### Loading the inputs and output files ####


text_array = []
text_file = open('/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/toy_data/txt.done.data')
for line in text_file:
  text_array.append(line.split('\n')[0]) #### all the sentences which are to be trained are  appended into an array and sent to the function of train Dataset for indices, lengths

wav_name=[]

folder = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/toy_data/wav'
train_set = train_database(text_array, folder) ###### folder containing wavefiles being sent to function train Dataset


######## Pytorch modules for training and loss calculation #########

model = Tacotron()
n_priority_freq = int(3000 / (sample_rate * 0.5) * num_freq)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss() # Least Absolute Deviations
####################################################################


for epoch in range(0,1):
 train_loader = DataLoader(dataset = train_set, batch_size=32, shuffle=True)

 for ig, (Text, Speech) in enumerate(train_loader):

###    print("text is", Text, numpy.shape(Text), "Speech is", numpy.shape(Speech))
    current_step = ig + epoch * len(train_loader) + 1

####### Preparing data for training ###########
    text = [text_to_seq(p) for p in Text]
    text = _prepare_data(text).astype(np.int32)

    wave =  [load_wav(p) for p in Speech]
    wave = _prepare_data(wave)
    magnitude = np.array([spectrogram(w) for w in wave])
    mel = np.array([melspectrogram(w) for w in wave])



######### This part needs some research ############

    timesteps = mel.shape[-1]
    if timesteps % outputs_per_step != 0: #### if loop when timsteps is not a multiple of 5 but why?
            magnitude = _pad_per_step(magnitude)
            mel = _pad_per_step(mel)

#####################################################
    mel_spec = mel ##### just making copies for future computations
    magnitude_spec = magnitude



    mel_input = np.concatenate((np.zeros([batch_size,num_mels, 1], dtype=np.float32),mel[:,:,1:]), axis=2)
#    print("mel input or decoder input is", numpy.shape(mel_input))
#   magnitude = _pad_per_step(magnitude)
    characters = Variable(torch.from_numpy(text).type(torch.LongTensor), requires_grad=False)  ## need to figure out why is requires grad false
    magnitude = Variable(torch.from_numpy(mel_input).type(torch.FloatTensor), requires_grad=False)
    magnitude_spec = Variable(torch.from_numpy(magnitude_spec).type(torch.Tensor), requires_grad=False)
    mel_spec = Variable(torch.from_numpy(mel).type(torch.Tensor), requires_grad=False)

    mel_output, linear_output = model.forward(characters, magnitude)
    #print("mag output", numpy.shape(mag_output), "linear is", numpy.shape(linear_output))

############ Training part #############################

    mel_loss = criterion(mel_output, mel_spec)
    linear_loss = torch.abs(linear_output-magnitude_spec)
    linear_loss = 0.5 * torch.mean(linear_loss) + 0.5 * torch.mean(linear_loss[:,:n_priority_freq,:])
    loss = mel_loss + linear_loss
###    loss = loss.cuda()
    print("loss is:", loss)
    start_time = time.time()

            # Calculate gradients
    loss.backward()

            # clipping gradients
    nn.utils.clip_grad_norm(model.parameters(), 1.)

            # Update weights
    optimizer.step()

    time_per_step = time.time() - start_time


####### When to print loss ################################


    if current_step % log_step == 0:
                print("time per step: %.2f sec" % time_per_step)
                print("At timestep %d" % current_step)
                print("linear loss: %.4f" % linear_loss.data[0])
                print("mel loss: %.4f" % mel_loss.data[0])
                print("total loss: %.4f" % loss.data[0])
