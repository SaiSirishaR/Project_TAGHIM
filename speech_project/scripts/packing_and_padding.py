#### explained with colour coding https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch ######

import torch
import numpy

a = [torch.tensor([1,2,3]), torch.tensor([3,4])]
print("a is", a)
#b = torch.nn.utils.rnn.pad_sequence(a,batch_first=False)
#print("after padding", b)
c = torch.nn.utils.rnn.pack_sequence(a)
print("after packing", c)
