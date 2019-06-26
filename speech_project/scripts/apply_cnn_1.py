## original  https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120/33
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class arctic_database(Dataset):

   def __init__(self, src):

    self.src = src

   def __len__(self):

    print("len is", len(self.src))
    return len(self.src)

   def __getitem__(self, idx):

    return self.src[idx]

########  padding, sorting and packing #####################

def pad_seq(sequence):
#  print("shape is", numpy.shape(sequence))
  ordered = sorted(sequence, key=len, reverse=True)
#  print("ordered is", ordered)
  lengths = [len(x) for x in ordered]
#  print("lenghts are", lengths)
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


def sort_batch(batch, lengths):
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    return seq_tensor,  seq_lengths 


Train_input=[]

input_folder = '/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/input_full'
os.chdir(input_folder)
input_files =sorted(os.listdir(input_folder))
for file in input_files:

   file_read = (numpy.loadtxt(file))
   Train_input.append(file_read)
#print("train inp is", len(Train_input))

Train_input = np.array(Train_input)
#print("after array thing", len(Train_input))
padded_input, lengths = pad_seq(Train_input)
#print("shape os padded input", numpy.shape(padded_input))
#inp,lens = sort_batch(torch.Tensor(padded_input), lengths)


eng_data = arctic_database(padded_input)

datass = DataLoader(eng_data, batch_size=4, shuffle=True)

for slt in datass:

    print("slt is", numpy.shape(slt.data), numpy.shape(slt.data.transpose(1,2)))

    slt.data = slt.data.transpose(1,2)
    m = nn.Conv1d(180, 10, 1) #60, 10,2  because of sstride the seq len varies
    input = Variable(torch.randn(4, 180, 1769))  # 9,60,953
    feature_maps1 = m(input)
    print("conv out shape is", numpy.shape(feature_maps1), numpy.shape(feature_maps1.transpose(1,2)))
    feature_maps1  = feature_maps1.transpose(1,2)
   #### packing for LSTM ######
    packed=rnn_utils.pack_sequence(feature_maps1)
    print("I have paked now", packed)
