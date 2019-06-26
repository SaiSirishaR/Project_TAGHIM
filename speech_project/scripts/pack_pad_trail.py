## original  https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120/33
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy

a = torch.Tensor([[1,2], [2,6], [3,4],[1,5]]) ## changed from 1 dim to 2
print("shape of a  is", numpy.shape(a), "a is", a)
b = torch.Tensor([[4,3], [5,5]])
print("shape osb is", numpy.shape(b))
c = torch.Tensor([[6,8]])
packed = rnn_utils.pack_sequence([a, b, c])
print("packed is", packed)
