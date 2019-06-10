from torch.utils.data import Dataset
import os 

class arctic_database(Dataset):

   def __init__(self, src_dir, tgt_dir):
    
    print("i got src dir as", src_dir)
    self.src_dir = src_dir
    self.tgt_dir = tgt_dir
    self.files = sorted(os.listdir(self.src_dir)) ### calling this in def __len__
 
   def __len__(self):

    print("len is", len(self.files))  #### __len__
    return len(self.files)

   def __getitem(self, idx): 

     return (sorted(os.listdir(self.src_dir))[idx])



eng_data = arctic_database(src_dir='/home/siri/backup_on8june/Documents/Projects/Interspeech_2019/Data/baseline_40_samples/input_full', 
                               tgt_dir='/home/siri/backup_on8june/Documents/Projects/Interspeech_2019/Data/baseline_40_samples/output_full')

#print("eng data len", [ p for p in eng_data[0]])
for i in range(len(eng_data)):

#    print("i is", i)
    sample = eng_data[i]
    print(sample)
