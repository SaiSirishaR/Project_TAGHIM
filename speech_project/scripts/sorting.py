## original  https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120/33
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy

a = torch.Tensor([[1,2,3],[3,4,2]]), torch.Tensor([[4,3,1],[2,6,1],[5,5,8]]), torch.Tensor([[6,8,4]]), torch.Tensor([[1,1,1]]) ## changed from 1 dim to 2
#lengths = [len(aa) for aa in a]
sorted =  sorted(a, key=len, reverse=True)
print("sorted", sorted)
packed=rnn_utils.pack_sequence(sorted)
print("packed", packed)
'''
print("length of a  is", len(a))
b = torch.Tensor([[4,3], [5,5]])
print("shape os b is", numpy.shape(b))
c = torch.Tensor([[6,8]])
print("shape os c is", numpy.shape(c))
packed = rnn_utils.pack_sequence(a)
print("packed is", packed)
lstm = nn.LSTM(3,3)

packed_output, (h,c) = lstm(packed)
print("packed out is", packed_output)
y = rnn_utils.pad_packed_sequence(packed_output)
print("y is", y)
'''
