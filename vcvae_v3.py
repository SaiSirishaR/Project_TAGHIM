from torch.utils.data import Dataset, DataLoader
import os
import numpy
from model_v1 import *
from torch import autograd
from torch.autograd import Variable
import numpy as np
from torch import nn, optim

batch_size = 4
input_dim = 60
output_dim = 60

class arctic_database(Dataset):

   def __init__(self, src_dir, tgt_dir):

    self.src_dir = src_dir
    self.tgt_dir = tgt_dir
    self.files = sorted(os.listdir(self.src_dir)) ### calling this in def __len__

   def __len__(self):

    return len(self.files)

   def __getitem__(self, idx):
    src_files = (sorted(os.listdir(self.src_dir)))
    tgt_files = (sorted(os.listdir(self.tgt_dir)))

    src_filenames = [os.path.join(self.src_dir,src_file) for src_file in src_files]
    tgt_filenames = [os.path.join(self.tgt_dir,tgt_file) for tgt_file in tgt_files]

    src_info = [numpy.loadtxt(src_filename) for src_filename in src_filenames]
    tgt_info =  [numpy.loadtxt(tgt_filename) for tgt_filename in tgt_filenames]

#    print("initial shape of src is", numpy.shape(src_info[idx]),"tgt shape  is", numpy.shape(tgt_info[idx]))
    return src_info[idx], tgt_info[idx] ## this is sending only one frame


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x





def collate_fn(batch):
    """Create batch"""
    r = 5 ### why is this
    input_lengths = [len(x[0]) for x in batch]
###    print("input lengths are:", input_lengths)
    max_input_len = np.max(input_lengths)
###    print("max_input_length is", max_input_len)
    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch]) + 1
###    print("max_tgt_length is", max_target_len)
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

#    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
#    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)

    b = np.array([_pad_2d(x[0], max_target_len) for x in batch],
                 dtype=np.float32)
    src_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    tgt_batch = torch.FloatTensor(c)
    return input_lengths, src_batch, tgt_batch



eng_data = arctic_database(src_dir='/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/mgc/input_full',
                               tgt_dir='/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/mgc/output_full')


vae_model = VAE()
optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss(size_average=False)


def loss_function(recon_x, x, mu, logvar):
    mse_loss = loss_fn(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - (mu ** 2) - torch.exp(logvar))
    return mse_loss + kld_loss



for epoch in range(0,1):
 datass = DataLoader(eng_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
 total_loss = 0
 for lengths, slt, bdl in datass:
  if numpy.shape(slt)[0]==batch_size:
   slt = np.array(slt)
   bdl = np.array(bdl)
   print("SF1 is", numpy.shape(slt.data), "TF1 is", numpy.shape(bdl.data))


   optimizer.zero_grad()
   dec_out, mu, var = vae_model(autograd.Variable(torch.Tensor(slt), requires_grad=False))
   dec_out = dec_out.view(batch_size,-1,output_dim)
   print("dec out shape is:", numpy.shape(dec_out), numpy.shape(dec_out.view(batch_size,-1,output_dim))) 
   loss = loss_function(dec_out, torch.Tensor(slt), mu, var)
   loss.backward()
   total_loss += loss.item()
   optimizer.step()
 print("epoch loss is:", epoch, total_loss/4)


####### Evaluation #########


class val_data(Dataset):

   def __init__(self, vall_dir):

    print("i got valid dir as", vall_dir)
    self.vall_dir = vall_dir
    self.files = sorted(os.listdir(self.vall_dir)) ### calling this in def __len__

   def __len__(self):

#    print("len is", len(self.files))  #### __len__
    return len(self.files)

   def __getitem__(self, idx):
#     print("am in get item")
#     print((sorted(os.listdir(self.src_dir))[idx]))
    valid_files = (sorted(os.listdir(self.vall_dir)))

    valid_filenames = [os.path.join(self.vall_dir,valid_file) for valid_file in valid_files]

    valid_info = src_info = [numpy.loadtxt(valid_filename) for valid_filename in valid_filenames]
    #print("am opening the file", "index",src_info[idx], "info", src_info)#numpy.loadtxt(filename))
 ####   print("initial shape", numpy.shape(src_info), "index is", numpy.shape(src_info[idx]))
    return valid_info[idx]


vcc_val_data = val_data(vall_dir='/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/mgc/valid_input')


val_datass = DataLoader(vcc_val_data, batch_size=1, shuffle=False)
for val in val_datass:
   print("inpt shape aftr datlaoding", numpy.shape(val), numpy.shape(val.view(1,-1,60)))
   val = np.array(val)
   print("inpt shape aftr arrayed", numpy.shape(val))
   val_out, val_mu, val_var = vae_model(autograd.Variable(torch.Tensor(val), requires_grad=False))
   print("val out shaope is", numpy.shape(val_out), numpy.shape(val_out.detach().numpy()))



