import torch
import torch.nn as nn
from symbols import symbols
import numpy
import numpy as np
from prenet import *
import torch.nn.functional as F
from torch import *
from torch.autograd import Variable
import random

embedding_size=256
hidden_size=128


num_mels = 80
num_freq = 1024
sample_rate = 16000
frame_length_ms = 50.
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
hidden_size = 128
embedding_size = 256

max_iters = 200
griffin_lim_iters = 60
power = 1.5
outputs_per_step = 5
teacher_forcing_ratio = 1.0

epochs = 10000
lr = 0.001
decay_step = [500000, 1000000, 2000000]
log_step = 100
save_step = 2000

class Encoder(nn.Module):

 def __init__(self, embedding_size):

   super(Encoder, self).__init__()
   self.embedding_size = embedding_size
   self.embed = nn.Embedding(len(symbols), embedding_size)
#   print("embed is", self.embed)
   self.prenet = prenet(embedding_size, hidden_size* 2, hidden_size)
   self.cbhg = CBHG(hidden_size)

 def forward(self, input_):

#        print("input shape initially is", input_, numpy.shape(input_)) #(4,61)
        input_ = torch.transpose(self.embed(input_),1,2) # transformed to ([4, 256, 61])
#        print("input shape after transpose", numpy.shape(input_))
        prenet = self.prenet.forward(input_)
#        print("received prenet info", numpy.shape(prenet))
        memory, hidden = self.cbhg.forward(prenet)
##        print("done with memory cbhg over prenet", numpy.shape(memory))
        return memory, hidden


class CBHG(nn.Module):
    """
    CBHG Module
    """
    def __init__(self, hidden_size, K=16, projection_size = 128, num_gru_layers=2, max_pool_kernel_size=2, is_post=False):
        """
        :param hidden_size: dimension of hidden unit
        :param K: # of convolution banks
        :param projection_size: dimension of projection unit
        :param num_gru_layers: # of layers of GRUcell
        :param max_pool_kernel_size: max pooling kernel size
        :param is_post: whether post processing or not
        """
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList()
        self.convbank_list.append(nn.Conv1d(in_channels=projection_size,
                                                out_channels=hidden_size,
                                                kernel_size=1,
                                                padding=int(np.floor(1/2))))

        for i in range(2, K+1):
            self.convbank_list.append(nn.Conv1d(in_channels=hidden_size,
                                                out_channels=hidden_size,
                                                kernel_size=i,
                                                padding=int(np.floor(i/2))))

        self.batchnorm_list = nn.ModuleList()
        for i in range(1, K+1):
            self.batchnorm_list.append(nn.BatchNorm1d(hidden_size))

        convbank_outdim = hidden_size * K
        if is_post == True:
 #           print("am in postprocessing conv1d shape is ", convbank_outdim, hidden_size*2)
            self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                             out_channels=hidden_size * 2,
                                             kernel_size=3,
                                             padding=int(np.floor(3/2)))
 #           print("am in postprocessing conv1d shape of proj2  is ", hidden_size*2, projection_size)
            self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size * 2,
                                               out_channels=projection_size,
                                               kernel_size=3,
                                               padding=int(np.floor(3/2)))
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size * 2)

        else:
            self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                             out_channels=hidden_size,
                                             kernel_size=3,
                                             padding=int(np.floor(3 / 2)))
            self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size,
                                               out_channels=projection_size,
                                               kernel_size=3,
                                               padding=int(np.floor(3 / 2)))
            self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size)

        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)


        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = nn.GRU(self.projection_size, self.hidden_size, num_layers=2,
                          batch_first=True,
bidirectional=True)

    def _conv_fit_dim(self, x, kernel_size=3):
        if kernel_size % 2 == 0:
            return x[:,:,:-1]
        else:
            return x


    def forward(self, input_):
#        print("input shape inside cbhg is", numpy.shape(input_))
        input_ = input_.contiguous()
#        print("input shape inside cbhaftr contigous g is", numpy.shape(input_))
        batch_size = input_.size()[0]

        convbank_list = list()
        convbank_input = input_
#        print("self conv bank lost is", self.convbank_list)
        # Convolution bank filters
        for k, (conv, batchnorm) in enumerate(zip(self.convbank_list, self.batchnorm_list)):
#            print("conv is", conv)
            convbank_input = F.relu(batchnorm(self._conv_fit_dim(conv(convbank_input), k+1).contiguous()))
            convbank_list.append(convbank_input)

        # Concatenate all features
        conv_cat = torch.cat(convbank_list, dim=1)

        # Max pooling
        conv_cat = self.max_pool(conv_cat)[:,:,:-1]

        # Projection
        conv_projection = F.relu(self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_cat))))
        conv_projection = self.batchnorm_proj_2(self._conv_fit_dim(self.conv_projection_2(conv_projection))) + input_
#        print("conv projection is", numpy.shape(conv_projection))
        # Highway networks
        highway = self.highway.forward(conv_projection)
        highway = torch.transpose(highway, 1,2)

        # Bidirectional GRU
        init_gru = Variable(torch.zeros(2 * self.num_gru_layers, batch_size, self.hidden_size))

        self.gru.flatten_parameters()
#        print("inut to gru n cbhg is", numpy.shape(highway))
        out, _ = self.gru(highway, init_gru)
#        print("cbhg out is", numpy.shape(out))
        return out, _


class Highwaynet(nn.Module):
    """
    Highway network
    """
    def __init__(self, num_units, num_layers=4):
        """
        :param num_units: dimension of hidden unit
        :param num_layers: # of highway layers
        """
        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = nn.ModuleList()
        self.linears = nn.ModuleList()
        for _ in range(self.num_layers):
            self.linears.append(SeqLinear(num_units, num_units))
            self.gates.append(SeqLinear(num_units, num_units))

    def forward(self, input_):

        out = input_

        # highway gated function
        for fc1, fc2 in zip(self.linears, self.gates):

            h = F.relu(fc1.forward(out))
            t = F.sigmoid(fc2.forward(out))

            c = 1. - t
            out = h * t + out * c
###        print("output of highway net is", out)
        return out


class AttentionDecoder(nn.Module):
    """
    Decoder with attention mechanism (Vinyals et al.)
    """
    def __init__(self, num_units):
        """
        :param num_units: dimension of hidden units
        """
        super(AttentionDecoder, self).__init__()
        self.num_units = num_units

        self.v = nn.Linear(num_units, 1, bias=False)
        self.W1 = nn.Linear(num_units, num_units, bias=False)
        self.W2 = nn.Linear(num_units, num_units, bias=False)

        self.attn_grucell = nn.GRUCell(num_units // 2, num_units)
        self.gru1 = nn.GRUCell(num_units, num_units)
        self.gru2 = nn.GRUCell(num_units, num_units)

        self.attn_projection = nn.Linear(num_units * 2, num_units)
        self.out = nn.Linear(num_units, num_mels * outputs_per_step)
    def forward(self, decoder_input, memory, attn_hidden, gru1_hidden, gru2_hidden):

        memory_len = memory.size()[1]
        batch_size = memory.size()[0]

        # Get keys
        keys = self.W1(memory.contiguous().view(-1, self.num_units))
        keys = keys.view(-1, memory_len, self.num_units)

        # Get hidden state (query) passed through GRUcell
        d_t = self.attn_grucell(decoder_input, attn_hidden)

        # Duplicate query with same dimension of keys for matrix operation (Speed up)
        d_t_duplicate = self.W2(d_t).unsqueeze(1).expand_as(memory)

        # Calculate attention score and get attention weights
        attn_weights = self.v(F.tanh(keys + d_t_duplicate).view(-1, self.num_units)).view(-1, memory_len, 1)
        attn_weights = attn_weights.squeeze(2)
        attn_weights = F.softmax(attn_weights)

        # Concatenate with original query
        d_t_prime = torch.bmm(attn_weights.view([batch_size,1,-1]), memory).squeeze(1)

        # Residual GRU
        gru1_input = self.attn_projection(torch.cat([d_t, d_t_prime], 1))
        gru1_hidden = self.gru1(gru1_input, gru1_hidden)
        gru2_input = gru1_input + gru1_hidden

        gru2_hidden = self.gru2(gru2_input, gru2_hidden)
        bf_out = gru2_input + gru2_hidden

        # Output
        output = self.out(bf_out).view(-1, num_mels, outputs_per_step)

        return output, d_t, gru1_hidden, gru2_hidden

    def inithidden(self, batch_size):
            attn_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False)
            gru1_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False)
            gru2_hidden = Variable(torch.zeros(batch_size, self.num_units), requires_grad=False)

            return attn_hidden, gru1_hidden, gru2_hidden




class MelDecoder(nn.Module):
    """
    Decoder
    """
    def __init__(self):
        super(MelDecoder, self).__init__()
        self.prenet = prenet(num_mels, hidden_size * 2, hidden_size)
        self.attn_decoder = AttentionDecoder(hidden_size * 2)

    def forward(self, decoder_input, memory):

        # Initialize hidden state of GRUcells
        attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.inithidden(decoder_input.size()[0])
        outputs = list()

        # Training phase
        if self.training:
            # Prenet
            dec_input = self.prenet.forward(decoder_input)
#            print("dec input shaoe is", numpy.shape(dec_input))
            timesteps = dec_input.size()[2] // outputs_per_step

            # [GO] Frame
            prev_output = dec_input[:, :, 0]
#            print("previous output is", numpy.shape(prev_output))
            for i in range(timesteps):
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.forward(prev_output, memory,
                                                                                             attn_hidden=attn_hidden,
                                                                                             gru1_hidden=gru1_hidden,
                                                                                             gru2_hidden=gru2_hidden)

                outputs.append(prev_output)

                if random.random() < 1.0:
                    # Get spectrum at rth position
                    prev_output = dec_input[:, :, i * 5]
                else:
                    # Get last output
                    prev_output = prev_output[:, :, -1]

            # Concatenate all mel spectrogram
            outputs = torch.cat(outputs, 2)
#            print("atende dec output is", numpy.shape(outputs))
        else:
            # [GO] Frame
            prev_output = decoder_input

            for i in range(max_iters):
                prev_output = self.prenet.forward(prev_output)
                prev_output = prev_output[:,:,0]
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.forward(prev_output, memory,
                                                                                         attn_hidden=attn_hidden,
                                                                                         gru1_hidden=gru1_hidden,
                                                                                         gru2_hidden=gru2_hidden)
                outputs.append(prev_output)
#                print("prev outpts initially", numpy.shape(prev_output))
                prev_output = prev_output[:, :, -1].unsqueeze(2)
#                print("prev outpts aftr sqeeze", numpy.shape(prev_output))
            outputs = torch.cat(outputs, 2)
#            print("outpts afr concat", numpy.shape(outputs))

        return outputs

class PostProcessingNet(nn.Module):
    """
    Post-processing Network
    """
    def __init__(self):
        super(PostProcessingNet, self).__init__()
#        print("am in postcbhg the hidden size is ", hidden_size)
        self.postcbhg = CBHG(hidden_size,
                             K=8,
                             projection_size=80,
                             is_post=True)
#        print("porcbhg is", self.postcbhg)
        self.linear = SeqLinear(hidden_size * 2,
                                1024)

    def forward(self, input_):
        out, _ = self.postcbhg.forward(input_)
#        print("out aftr postcbhg is", numpy.shape(out))
        out = self.linear.forward(torch.transpose(out,1,2))

        return out



class Tacotron(nn.Module):

  def __init__(self):

    super(Tacotron, self).__init__()
    self.encoder = Encoder(embedding_size)
    self.decoder1 = MelDecoder()
    self.decoder2 = PostProcessingNet()

  def forward(self, characters, mel_input):
        memory, hidden  = self.encoder.forward(characters)
#        print("memory is", numpy.shape(memory), "hidden is", numpy.shape(hidden))
#        print("dec inp is", numpy.shape(mel_input))
        mel_output = self.decoder1.forward(mel_input, memory)
#        print("got mel_output", numpy.shape(mel_output))
        linear_output = self.decoder2.forward(mel_output)
#        print("linear output after post processing", numpy.shape(linear_output))
        return mel_output, linear_output
