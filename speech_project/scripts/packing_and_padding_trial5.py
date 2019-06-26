## original  https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120/33
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy
import os
import numpy as np


######## Sorting without padding #####################

Train_input=[]

input_folder = '/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/input_full'
os.chdir(input_folder)
input_files =sorted(os.listdir(input_folder))
for file in input_files:

   file_read = torch.Tensor(numpy.loadtxt(file))
   Train_input.append(file_read)
print("train inp is", len(Train_input))

sorted =  sorted(Train_input, key=len, reverse=True)
print("sorted", numpy.shape(sorted[2]))

#################### Sorted ########################

##### Packing the sorted seq #################

packed=rnn_utils.pack_sequence([d for d in sorted])
print("packed",packed)
