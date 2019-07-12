############## This code is Encoder Decoder Model until Training, writes predcited feats into files, SF1-TF1 conversion ################## (there is only padding, need to add sort and pack)

import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy
import os, sys
import numpy as np
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from seq2seq_net_attend_v1 import *
import torch.optim as optim

class arctic_database(Dataset):

   def __init__(self, src, tgt):

    self.src = src
    self.tgt = tgt

   def __len__(self):

##    print("len is", len(self.src))
    return len(self.src)

   def __getitem__(self, idx):

    return self.src[idx], self.tgt[idx]

########  padding, sorting and packing #####################

def pad_seq(sequence):
  ordered = sorted(sequence, key=len, reverse=True)
  lengths = [len(x) for x in ordered]
  max_length = lengths[0]
  seq_len = [len(seq) for seq in sequence]
  padded = []
  for i in range(0,len(sequence)):
   npad = ((0, max_length-len(sequence[i])),(0,0))
   padded.append(np.pad(sequence[i], pad_width=npad, mode='constant', constant_values = 0))
  return padded, lengths



def pad_seq_valid(sequence):

  ordered = sorted(sequence, key=len, reverse=True)
  lengths = [len(x) for x in ordered]
  lengths_v = [len(x) for x in sequence]
  max_length = lengths[0]
  seq_len = [len(seq) for seq in sequence]
  #print("seq len is:", seq_len)
  padded = []
  for i in range(0,len(sequence)):
   npad = ((0, max_length-len(sequence[i])), (0,0))
   padded.append(np.pad(sequence[i], pad_width=npad, mode='constant', constant_values = 0))
  return padded, lengths_v


def sort_batch(batch, lengths):
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    return seq_tensor,  seq_lengths 


Train_input=[]
input_folder = '/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/mgc/input_full'
os.chdir(input_folder)
input_files =sorted(os.listdir(input_folder))
for file in input_files:

   file_read = (numpy.loadtxt(file))
   Train_input.append(file_read)

Train_input = np.array(Train_input)



Train_output=[]
output_folder = '/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/mgc/output_full'
os.chdir(output_folder)
output_files =sorted(os.listdir(output_folder))
for out_file in output_files:

   out_file_read = (numpy.loadtxt(out_file))
   Train_output.append(out_file_read)

Train_output = np.array(Train_output)



padded_input, lengths = pad_seq(Train_input)
padded_output, lengths = pad_seq(Train_output)

eng_data = arctic_database(torch.Tensor(padded_input), torch.Tensor(padded_output))

datass = DataLoader(eng_data, batch_size=4, shuffle=True)

loss_fn = torch.nn.MSELoss(size_average=False)


enc_input_dim = 60
enc_hidden_dim =256
dec_output_dim= 60

encdec_model = EncDec(enc_input_dim, enc_hidden_dim, dec_output_dim)
print(encdec_model)
encoder_hidden = encdec_model.enc.initHidden(4)


optimizer = optim.Adagrad(encdec_model.parameters(), lr=0.01, lr_decay=0, weight_decay=0)


for i in range(0,10):
 for SF1, TF1 in datass:
   if SF1.data.transpose(0,1).shape[1] == 4:    
    input_to_enc = SF1.data.transpose(0,1) # shape is seq_len, batch, input_dim
    a = input_to_enc[:,:,1:]

    input_to_dec = TF1.data.transpose(0,1)
    b = input_to_dec[:,:,1:]
    padding =np.zeros([numpy.shape(b)[0],numpy.shape(b)[1],197])
    new_b =  np.concatenate((b, padding), axis=2)

    zero_tensor = np.zeros([TF1.data.transpose(0,1).shape[0],4,1])
    dec_input = np.concatenate((zero_tensor, new_b[:,:,1:]), axis=2)

    linear_out, dec_out = encdec_model(autograd.Variable(input_to_enc), encoder_hidden, autograd.Variable(torch.Tensor(dec_input)))    

'''
##### Loss calculation #######

    loss = loss_fn(linear_out.transpose(0,1), TF1.data)
    #print("loss is", loss.data[0]/4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 print("epoch los for epoch:", i, loss.data[0]/4)




######## Evaluation #########

pred_dir = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Project_TAGHIM/speech_project/scripts/eval_pred'

valid_input = []
valid_filename=[]
valid_folder = '/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/mgc/valid_input'
os.chdir(valid_folder)
valid_files =sorted(os.listdir(valid_folder))
for file in valid_files:
   valid_filename.append(file)
   print(file)
   p = numpy.loadtxt(file)
   valid_input.append(p)

valid_input = np.array(valid_input)

padded_val, lengths_val = pad_seq_valid(valid_input)



val_data = arctic_database(torch.Tensor(padded_val), torch.Tensor(padded_val))

datass_val = DataLoader(val_data, batch_size=1, shuffle=False)

for g, (inp, tmp) in enumerate(datass_val):

    d = open(pred_dir+'/'+valid_filename[g].split('.')[0]+'.mgc_ascii','w')

    #print("len of inp is", len(inp), numpy.shape(inp))
    eval_enc_hidden = Variable(torch.zeros(1,1, enc_hidden_dim))

    enc_output, eval_enc_hidden = encdec_model.enc.forward(autograd.Variable(torch.Tensor(inp.data.transpose(0,1))), eval_enc_hidden)

    eval_dec_input = np.zeros([1,1,enc_hidden_dim])

    final_dec_output = []

    for jk in range(0,numpy.shape(inp)[1]):
     if jk < lengths_val[g]:   
      linear_out_eval, dec_output_eval = encdec_model.dec.forward(autograd.Variable(torch.Tensor(eval_dec_input)), eval_enc_hidden)
      for hp in range(0,len(linear_out_eval.transpose(0,1).data[0].detach().numpy())):
        for kp in range(0,len(linear_out_eval.transpose(0,1).data[0].detach().numpy()[hp])):
          d.write(str(linear_out_eval.transpose(0,1).data[0].detach().numpy()[hp][kp]) +' ')
        d.write('\n')
      eval_dec_input = dec_output_eval
    d.close()
        
    print("wrote", valid_filename[g])     

'''
