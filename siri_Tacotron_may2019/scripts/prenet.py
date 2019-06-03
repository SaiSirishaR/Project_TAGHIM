import torch
import os
import torch.nn as nn
import numpy as np

input_size = 256
output_size= 1024
hidden_size = 128

class prenet(nn.Module):

 def __init__(self, input_size, hidden_size, output_size):

   super(prenet, self).__init__()
   self.input_size = input_size
   self.output_size = output_size
   self.hidden_size = hidden_size
   self.layer = nn.Sequential(
             (nn.Linear(self.input_size, self.hidden_size)),
             (nn.ReLU()),
             ('dropout1', nn.Dropout(0.5)),
             ('fc2', nn.Linear(self.hidden_size, self.output_size)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(0.5)),)

 def forward(self, input_):

        out = self.layer(input_)
        print("out is", out)
        return out
  

###file = open('/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Data/txt.done.data')
###for line in file:
###    print("data is", line)



#prenet = prenet(input_size, hidden_size*2, hidden_size )
#prenet = self.prenet.forward(input_)
