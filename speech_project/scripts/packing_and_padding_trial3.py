### Takes 100 samples as batch ####### experiment with this

## original  https://discuss.pytorch.org/t/simple-working-example-how-to-use-packing-for-variable-length-sequence-inputs-for-rnn/2120/33
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy
import os
import numpy as np




#### Trail 3 #########

'''
train_input = []

input_files = [filename for filename in sorted(os.listdir('/home3/srallaba/projects/siri_expts/vocal_loudness_expts/Data/stage_1_feats/input_full'))]
for input_file in input_files:
 A = np.loadtxt('/home3/srallaba/projects/siri_expts/vocal_loudness_expts/Data/stage_1_feats/input_full/' + input_file)

 for a in A:
   train_input.append(a)

train_input = np.array(train_input)
p = torch.Tensor(train_input)
packed=rnn_utils.pack_sequence(p)
print("packed", packed)
'''

#### Trail 2#####


'''
input_folder = '/home3/srallaba/projects/siri_expts/vocal_loudness_expts/Data/stage_1_feats/input_full'
input_files =sorted(os.listdir(input_folder))
loaded_files =  [torch.Tensor(numpy.loadtxt(input_folder+'/'+file)) for file in input_files]
loaded_files = np.array(loaded_files)
print("am getting this", numpy.shape(loaded_files[0]))
packed=rnn_utils.pack_sequence(loaded_files)
print("packed")#, numpy.shape(packed))
print(packed)



'''


#### Trail 1 #####



'''
a = torch.Tensor([[1,2], [2,6], [3,4]]), torch.Tensor([[4,3], [5,5]]), torch.Tensor([[6,8]]) ## changed from 1 dim to 2
print("length of a  is", len(a))
#b = torch.Tensor([[4,3], [5,5]])
#c = torch.Tensor([[6,8]])
packed = rnn_utils.pack_sequence(a)
print("packed is", packed)
lstm = nn.LSTM(2,3)

packed_output, (h,c) = lstm(packed)
print("packed out is", packed_output)
y = rnn_utils.pad_packed_sequence(packed_output)
print("y is", y)
'''

Train_input=[]

input_folder = '/home3/srallaba/projects/siri_expts/vocal_loudness_expts/Data/stage_1_feats/input_full'
os.chdir(input_folder)
input_files =sorted(os.listdir(input_folder))
for file in input_files:

   file_read = torch.Tensor(numpy.loadtxt(file))
   Train_input.append(file_read)
print("train inp is", len(Train_input))
new_train_data  = torch.nn.utils.rnn.pad_sequence(Train_input, batch_first=True)
print("adat is padded", numpy.shape(new_train_data))
#sorted = lengths.sort(Train_input, decreasing=True)
packed=rnn_utils.pack_sequence(new_train_data)
print("packing done", packed)
