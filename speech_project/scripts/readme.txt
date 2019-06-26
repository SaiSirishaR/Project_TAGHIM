
###### Day wise tasks #####

June 8

Data loading and retreving
resolved len issue

June 9

resolved get item issue

June 10

Reading the wavefiles in the function get item
Added dataloader
index thing--> 671*60 and 761*60 are stored as 60,1 which means stores info column wise (dimension wise) =====> apparently this is why i did padding before calling DataLoader in my LSTM code
DataLoader --> does batching and shuffling --> https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

1) return index --> ur input shape is changed
2) use dictionary DataLoader needs tyhe input of same shape

Task to do tranform the matrix shape before feeding it to the DataLoader

June 16

need to transfrom matrix into a 3 Dim one and apply conv1D
explained packing and padding with colour coding --> https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch

flattening seqs by keeping track of batch size at each timestep.

June 17

1) torch.nn.utils.rnn has padding and packing

2) need not pad just pack --> https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120/24

3) padding github example with char seq --> https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial

4) pad_packed_seq is nothing but unpack --> https://discuss.pytorch.org/t/rnn-with-pack-and-unpack-is-significantly-slower/2165

5) batch indicates number of elemenst or len of seq in bacth_first=true

6) whats the difference between torch.tensor and torch.Tensor??

7) when used torch.tensor gave torch.Float required got torch.Long but it was solved when used torch.Tensor

8) the pack function assumes the inputs seq lengths are in decresing order, so sort the sequences before feeding it to packing

Leaving for meeting ToDo--> sorting

In order to sort you need to pad and obtain lengths

9) sorted order stored and can be retrived --> https://github.com/pytorch/pytorch/issues/3584

Currently, pack_padded_sequence requires the caller to sort inputs by the length (descending). We should sort and reorder the input internally if it's not already sorted. We can then restore the original order in pad_packed_sequence.

10) see pack_pad.py code....its doing this

4 3 2 1 --> batch
1 2 3 --> batch
1 2
1


so i ned to do all the three padding, sorting, packing


18 June 2019

Planning --> 

1) go to dataset class 2) transfrom 3) dataloader

2-transform has 3 things

 i) pad, ii) sort, iii) pack
 
Dataset class at a time gives me a batch

np.pad ((a,b),(c,d))
a -> adds zeros in the begging of the row
b-> adds zaros at the end of the row
c-> adds an other dim on the left
d-> adds an other dim at right



conv layr -> embed size or dim, out_channels, stride
input to conv -> batch, dim, seq_len


june 20, 2019


for packing u need lengths for which you are giving lengths inside the dataloader ->  dataloading_padding18june_v1.py


found this:

padding+sorting+packing = sorting+packing
