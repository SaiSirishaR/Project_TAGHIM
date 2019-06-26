## original  https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120/33
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy
import os
import numpy as np


########  Padding thing is here #####################

Train_input=[]

input_folder = '/home3/srallaba/projects/siri_expts/vocal_loudness_expts/Data/stage_1_feats/temp'
os.chdir(input_folder)
input_files =sorted(os.listdir(input_folder))
for file in input_files:

   file_read = torch.Tensor(numpy.loadtxt(file))
   Train_input.append(file_read)
print("train inp is", len(Train_input))
new_train_data  = torch.nn.utils.rnn.pad_sequence(Train_input, batch_first=True)
print("padding on", numpy.shape(new_train_data[2]), "orig", numpy.shape(Train_input[2]), numpy.shape(new_train_data))

####### Padding is done here sort now ########


####### Sorting thing #############

##print("adat is padded", numpy.shape(new_train_data))
#sorted = lengths.sort(Train_input, decreasing=True)

##packed=rnn_utils.pack_sequence(new_train_data)
##print("packing done", packed)
