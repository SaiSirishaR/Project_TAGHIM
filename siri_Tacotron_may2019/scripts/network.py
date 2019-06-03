import torch
from symbols import symbols
from prenet import prenet
import torch.nn as nn
import numpy

class Encoder(nn.Module):
   
 def __init__(self, embedding_size):

   super(Encoder, self).__init__()
   self.embedding_size = embedding_size
   self.embed = nn.Embedding(len(symbols), embedding_size)
#   self.prenet = prenet(embedding_size, 128 * 2, 128)
#   self.cbhg = CBHG(128)
   print("self embed", self.embed)
#   print("self.prenet", self.prenet)

 def forward(self, input_):

        input_ = torch.transpose(self.embed(input_),1,2)
        prenet = self.prenet.forward(input_)
        memory = self.cbhg.forward(prenet)

        return memory


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
        if is_post:
            self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                             out_channels=hidden_size * 2,
                                             kernel_size=3,
                                             padding=int(np.floor(3/2)))
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
        print("output of highway net is", out)
        return out




class Tacotron(nn.Module):

  def __init__(self):

    super(Tacotron, self).__init__()
    self.encoder= Encoder(256)
#    self.decoder1 = Decoder()
#    self.decoder2 = PostProcessingNet()

   
  def forward(self, characters, mel_input):
        memory = self.encoder.forward(characters)
        mel_output = self.decoder1.forward(mel_input, memory)
        linear_output = self.decoder2.forward(mel_output)
        return mel_output, linear_output
