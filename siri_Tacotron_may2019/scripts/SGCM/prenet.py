import torch
import torch.nn as nn
from collections import OrderedDict
import numpy
import numpy as np

input_size = 256
output_size= 1024
hidden_size = 128



class SeqLinear(nn.Module):
    """
    Linear layer for sequences
    """
    def __init__(self, input_size, output_size, time_dim=2):
        """
        :param input_size: dimension of input
        :param output_size: dimension of output
        :param time_dim: index of time dimension
        """
        super(SeqLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.time_dim = time_dim
        self.linear = nn.Linear(input_size, output_size)


    def forward(self, input_):
        """
        :param input_: sequences
        :return: outputs
        """
        batch_size = input_.size()[0]
        
        if self.time_dim == 2:
###            print("yes dim is 2 am in loop", numpy.shape(input_))
            input_ = input_.transpose(1, 2).contiguous()
###            print("after transpose", numpy.shape(input_))
        input_ = input_.view(-1, self.input_size)
###        print("view statement", numpy.shape(input_))

        out = self.linear(input_).view(batch_size, -1, self.output_size)
###        print("out shaps is", numpy.shape(out),"aftr linear it was", numpy.shape(self.linear(input_)))
        if self.time_dim == 2:
            out = out.contiguous().transpose(1, 2)
###            print("after transpose", numpy.shape(out))
        return out

class prenet(nn.Module):

 def __init__(self, input_size, hidden_size, output_size):

   super(prenet, self).__init__()
   self.input_size = input_size
   self.output_size = output_size
   self.hidden_size = hidden_size
   self.layer = nn.Sequential(OrderedDict([
             ('fc1', SeqLinear(self.input_size, self.hidden_size)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(0.5)),
             ('fc2', SeqLinear(self.hidden_size, self.output_size)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(0.5)),
]))

 def forward(self, input_):

        out = self.layer(input_)
        print("prenet out is",  numpy.shape(out))
        return out

