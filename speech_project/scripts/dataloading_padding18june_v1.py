## original  https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120/33
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import autograd
from torch.autograd import Variable

class arctic_database(Dataset):

   def __init__(self, src, lengths):

    self.src = src
    self.len = lengths

   def __len__(self):

#    print("len is", len(self.src))
    return len(self.src)

   def __getitem__(self, idx):

    return self.src[idx],  self.len[idx]

########  padding, sorting and packing #####################

def pad_seq(sequence):
#  print("shape is", numpy.shape(sequence))
  ordered = sorted(sequence, key=len, reverse=True)
  lengths = [len(x) for x in ordered]
  max_length = lengths[0]
  seq_len = [len(seq) for seq in sequence]
 # print("seq len is:", seq_len)
  padded = []
  for i in range(0,len(sequence)):
   npad = ((0, max_length-len(sequence[i])), (0,0))
#   print("n pad shape is", numpy.shape(npad), npad)
   padded.append(np.pad(sequence[i], pad_width=npad, mode='constant', constant_values = 0))
  return padded, lengths


def sort(batch, lengths):
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


eng_data = arctic_database(padded_input, lengths)

datass = DataLoader(eng_data, batch_size=4, shuffle=True)

for slt,lengths in datass:

    print("slt is", numpy.shape(slt.data), "lenghs are", lengths)

    slt.data, lengths = sort(slt.data, lengths)
    print("aftr sorting", lengths)
    pack = torch.nn.utils.rnn.pack_padded_sequence(autograd.Variable(slt), lengths, batch_first=True)
    print("i have packed the sea", pack)
