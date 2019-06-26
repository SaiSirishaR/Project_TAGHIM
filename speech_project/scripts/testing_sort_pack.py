import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy

'''

a = torch.Tensor(numpy.loadtxt('/home3/srallaba/projects/siri_expts/vocal_loudness_expts/Data/stage_1_feats/temp/arctic_a0001_aligned.coeffs'))
print("a shape is", numpy.shape(a))
b = torch.Tensor(numpy.loadtxt('/home3/srallaba/projects/siri_expts/vocal_loudness_expts/Data/stage_1_feats/temp/arctic_a0001_aligned.coeffs'))
print("b is", numpy.shape(b))
c = torch.Tensor(numpy.loadtxt('/home3/srallaba/projects/siri_expts/vocal_loudness_expts/Data/stage_1_feats/temp/arctic_a0001_aligned.coeffs'))

packed = rnn_utils.pack_sequence([a, b, c])
print("packed", packed)



a = torch.Tensor([[1,2], [2,6], [3,4]]) ## changed from 1 dim to 2
print("shape of a  is", numpy.shape(a), "a is", a)
b = torch.Tensor([[4,3], [5,5]])
print("shape osb is", numpy.shape(b))
c = torch.Tensor([[6,8]])
packed = rnn_utils.pack_sequence([a, b, c])
print("packed is", packed)

'''




aline_array = []
a = open('/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/input_full/10013_aligned.coeffs')
for line in a:

   line = line.split('\n')[0].strip(' ')
   aline_array.append(line)

'''
#print("wat i got", len(line_array))
bline_array = []
b = open('/home3/srallaba/projects/siri_expts/vocal_loudness_expts/Data/stage_1_feats/temp/arctic_a0002_aligned.coeffs')


for bline in b:

   bline = bline.split('\n')[0].strip(' ')
   bline_array.append(bline)
'''
packed = rnn_utils.pack_sequence([torch.Tensor(aline_array)])

