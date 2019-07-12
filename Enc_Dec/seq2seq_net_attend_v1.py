import torch
import torch.nn as nn
import numpy
import torch.autograd
from torch.autograd import Variable
import math
import torch.nn.functional as F


class Attention(nn.Module):
  
    def __init__(self, enc_hidden_dim):

      super(Attention, self).__init__()
      self.enc_hidden_dim = enc_hidden_dim
      self.scale = 1. / math.sqrt(self.enc_hidden_dim)


    def forward(self, hidden_state, enc_output, enc_output1):

       hidden_state = hidden_state.transpose(0,1).transpose(1,2) # [b,1,dim] to [b,dim,1]
       print("hidden state is", numpy.shape(hidden_state))   
       enc_output = enc_output.transpose(0,1) #[seq,b,dim] to [b,seq,dim]
       print("enc out shape is", numpy.shape(enc_output))


       energy = torch.bmm(enc_output, hidden_state)# [b,seq,1]
       print("ebergy shape is", numpy.shape(energy))
       energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize
       print("energy shape is", numpy.shape(energy)) #[b,seq,1]
       print("enc out is", numpy.shape(enc_output1))
       enc_output1 = enc_output1.transpose(0,1) #[seq,b,dim] to [b,seq,dim]
       print("enc output is", numpy.shape(enc_output1), numpy.shape(enc_output1.transpose(1,2)))
       print("enc out later  is", numpy.shape(enc_output1.transpose(1,2)), numpy.shape(energy.transpose(1,2).transpose(1,2)))
       linear_combination = torch.bmm(enc_output1.transpose(1,2), energy.transpose(1,2).transpose(1,2))
       print("l comb is", numpy.shape(linear_combination.squeeze())) #b,1,dim
       return energy, linear_combination




class Encoder(nn.Module):

 def __init__(self, input_dim, hidden_dim):

   super(Encoder, self).__init__()
   self.input_dim = input_dim
   self.hidden_dim = hidden_dim
   self.gru = nn.GRU(input_dim, hidden_dim, 1) 


 def forward(self, input_, h0):
        
        #h0 = self.initHidden(numpy.shape(input_)[1])        
###        print("inputs is", numpy.shape(input_), "h0 is", numpy.shape(h0))
        output, hidden = self.gru(input_, h0)
###        print("output is", numpy.shape(output), "hidden is", numpy.shape(hidden))
#        return memory
        return output, hidden

 def initHidden(self,batch_size):
        result = Variable(torch.zeros(1,batch_size , self.hidden_dim)) #layers*directions, batch, hidden_dim)
###        print("result is", numpy.shape(result), "result is", result)
        return result



class Decoder(nn.Module):

 def __init__(self, dec_input_dim, dec_hidden_dim, dec_output_dim):

   super(Decoder, self).__init__()
   self.dec_input_dim = dec_input_dim
   self.dec_hidden_dim = dec_hidden_dim
   self.dec_output_dim = dec_output_dim
   self.dec_gru = nn.GRU(self.dec_input_dim, self.dec_hidden_dim, 1)
   self.linear = nn.Linear(self.dec_hidden_dim, self.dec_output_dim)
   self.attend = Attention(dec_input_dim) # encoder hidden dim 

 def forward(self, dec_input_, dec_h0):

        energy, linear_comb = self.attend(dec_h0, dec_input_, dec_input_)
        print("energy sghap", numpy.shape(energy), "linear comb shape:", numpy.shape(linear_comb))
        print("try this", energy.squeeze()(linear_comb))
        #print("dec inputs is", numpy.shape(torch.), "h0 is", h0)

        dec_ou = self.linear(linear_comb)
        print("dec output is=============> ", numpy.shape(dec_ou))
        return dec_ou
'''
        dec_output, dec_hidden = self.dec_gru(dec_input_, dec_h0)
  
###        print("dec output is", numpy.shape(dec_output), "dec.view(-1,256 iss)", numpy.shape(dec_output.view(-1,self.dec_hidden_dim)), "dec hidden is", numpy.shape(dec_hidden))
        print("dec input shpe is", numpy.shape(dec_output))
        linear_out = self.linear(dec_output)
        print("linear ot is", numpy.shape(linear_out))
        return linear_out, dec_output
#        return dec_output, dec_hidden
'''

class EncDec(nn.Module):

    def __init__(self, enc_input_dim, enc_hidden_dim, dec_output_dim):

         super(EncDec, self).__init__()
         self.enc = Encoder(enc_input_dim, enc_hidden_dim)
         self.dec = Decoder(enc_hidden_dim, enc_hidden_dim, dec_output_dim)         
         
    def forward(self, enc_input,  enc_hidden, dec_input):

       enc_out, enc_hn = self.enc(enc_input, enc_hidden)
###       print("ran encoder")
       decoder_h0 = enc_hn
###       print("dec hidden si ready")
       dec_out = self.dec(dec_input, decoder_h0)
       return  dec_out



    
