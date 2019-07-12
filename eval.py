from torch.utils.data import Dataset, DataLoader
import os
import numpy
from model_v1 import *
from torch import autograd
from torch.autograd import Variable
import numpy as np
from torch import nn, optim

class arctic_database(Dataset):

   def __init__(self, src_dir):

  ####  print("i got src dir as", src_dir)
    self.src_dir = src_dir
    self.files = sorted(os.listdir(self.src_dir)) ### calling this in def __len__

   def __len__(self):

#    print("len is", len(self.files))  #### __len__
    return len(self.files)

   def __getitem__(self, idx):
#     print("am in get item")
#     print((sorted(os.listdir(self.src_dir))[idx]))
    src_files = (sorted(os.listdir(self.src_dir))[idx])

    src_filename = os.path.join(self.src_dir,src_files)
 ####   print("source filename is", src_filename)
 ####   print("target filename is", tgt_filename)

    src_info = numpy.loadtxt(src_filename)
    #print("am opening the file", "index",src_info[idx], "info", src_info)#numpy.loadtxt(filename))
 ####   print("initial shape", numpy.shape(src_info), "index is", numpy.shape(src_info[idx]))
    return src_info[idx]
 #     return (sorted(os.listdir(self.src_dir))[idx])



eng_data = arctic_database(src_dir='/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/mgc/valid_input')





'''
from torch.utils.data import Dataset, DataLoader
import os
import numpy
from model_v1 import *
from torch import autograd
from torch.autograd import Variable
import numpy as np
from torch import nn, optim


class val_data(Dataset):

   def __init__(self, valid_dir):

    print("i got valid dir as", valid_dir)
    self.valid_dir = valid_dir
    self.files = sorted(os.listdir(self.valid_dir)) ### calling this in def __len__

   def __len__(self):

#    print("len is", len(self.files))  #### __len__
    return len(self.files)

   def __getitem__(self, idx):
#     print("am in get item")
#     print((sorted(os.listdir(self.src_dir))[idx]))
    valid_files = (sorted(os.listdir(self.valid_dir))[idx])

    valid_filename = os.path.join(self.valid_dir,valid_files)
 ####   print("target filename is", tgt_filename)

    valid_info = numpy.loadtxt(valid_filename)
    #print("am opening the file", "index",src_info[idx], "info", src_info)#numpy.loadtxt(filename))
 ####   print("initial shape", numpy.shape(src_info), "index is", numpy.shape(src_info[idx]))
    return valid_info[idx]



vcc_val_data = val_data(vall_dir='/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/mgc/valid_input')


val_datass = DataLoader(vcc_val_data, batch_size=1, shuffle=True)
for val in val_datass:
   val = np.array(val)
   val_out, val_mu, val_var = vae_model(autograd.Variable(torch.Tensor(val), requires_grad=False))
   print("val out shaope is", numpy.shape(val_out), numpy.shape(val_out.numpy()))

'''
