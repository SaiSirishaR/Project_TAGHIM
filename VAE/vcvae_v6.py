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

    src_info = numpy.loadtxt(src_filenames[idx])
    tgt_info = numpy.loadtxt(tgt_filenames[idx])

    return src_info, tgt_info 


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x


def collate_fn(batch):
    """Create batch"""
    r = 5 ### why is this
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths)
    max_target_len = np.max([len(x[1]) for x in batch]) + 1

    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

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
#optimizer = optim.Adam(vae_model.parameters(), lr=1e-2)
optimizer = optim.Adagrad(vae_model.parameters(), lr=0.01, lr_decay=0, weight_decay=0)
loss_fn = torch.nn.MSELoss(size_average=False)


def loss_function(recon_x, x, mu, logvar):
    mse_loss = loss_fn(recon_x, x)
    kld_loss = -0.5 * torch.sum(1 + logvar - (mu ** 2) - torch.exp(logvar))
    return mse_loss + kld_loss


for epoch in range(0,10):
 datass = DataLoader(eng_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
 total_loss = 0
 count =0 #https://github.com/atinghosh/VAE-pytorch/blob/master/simple_main.py
 for lengths, slt, bdl in datass:
  if numpy.shape(slt)[0]==batch_size:
   count += slt.size(0)

######### Training the VAE ########

   optimizer.zero_grad()
   dec_out, mu, var = vae_model(autograd.Variable(torch.Tensor(slt.data), requires_grad=False))
   loss = loss_function(dec_out, torch.Tensor(slt), mu, var)
   loss.backward()
   total_loss += loss.item()
   optimizer.step()
 total_loss /= count
 print("count is:", count)
 print("epoch loss is:", epoch, total_loss)


####### Evaluation #########
pred_dir = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Project_TAGHIM/siri_Tacotron_may2019/scripts/SGCM/VAE/VAE_predictions'

class val_data(Dataset):

   def __init__(self, vall_dir):

    print("i got valid dir as", vall_dir)
    self.vall_dir = vall_dir
    self.files = sorted(os.listdir(self.vall_dir)) ### calling this in def __len__

   def __len__(self):
    return len(self.files)

   def __getitem__(self, idx):
    valid_files = (sorted(os.listdir(self.vall_dir)))
    valid_filenames = [os.path.join(self.vall_dir,valid_file) for valid_file in valid_files]
    valid_info = numpy.loadtxt(valid_filenames[idx])
    return valid_info, valid_files[idx]


vcc_val_data = val_data(vall_dir='/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/mgc/valid_input')


val_datass = DataLoader(vcc_val_data, batch_size=1, shuffle=False)
for val, fname in val_datass:
   val = np.array(val)
   val_out, val_mu, val_var = vae_model(autograd.Variable(torch.Tensor(val), requires_grad=False))
   np.savetxt(pred_dir + '/'+ fname[0], val_out.data[0].detach().numpy())



