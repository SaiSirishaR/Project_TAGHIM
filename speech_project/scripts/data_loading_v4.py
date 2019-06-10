from torch.utils.data import Dataset, DataLoader
import os 
import numpy

class arctic_database(Dataset):

   def __init__(self, src_dir, tgt_dir):
    
    print("i got src dir as", src_dir)
    self.src_dir = src_dir
    self.tgt_dir = tgt_dir
    self.files = sorted(os.listdir(self.src_dir)) ### calling this in def __len__
 
   def __len__(self):

#    print("len is", len(self.files))  #### __len__
    return len(self.files)

   def __getitem__(self, idx): 
#     print("am in get item")
#     print((sorted(os.listdir(self.src_dir))[idx]))
    src_files = (sorted(os.listdir(self.src_dir))[idx])
    tgt_files = (sorted(os.listdir(self.tgt_dir))[idx])

    src_filename = os.path.join(self.src_dir,src_files)   
    print("source filename is", src_filename)
    tgt_filename = os.path.join(self.tgt_dir,tgt_files)
    print("target filename is", tgt_filename)
    
    src_info = numpy.loadtxt(src_filename)
    tgt_info = numpy.loadtxt(tgt_filename)
    #print("am opening the file", "index",src_info[idx], "info", src_info)#numpy.loadtxt(filename))
    print("initial shape", numpy.shape(src_info), "index is", numpy.shape(src_info[idx]))
    sample = {'source_slt': src_info, 'target_bdl': tgt_info}
#    return src_info[idx], tgt_info[idx]
 #     return (sorted(os.listdir(self.src_dir))[idx])

    return sample

eng_data = arctic_database(src_dir='/home/siri/backup_on8june/Documents/Projects/Interspeech_2019/Data/baseline_40_samples/input_full', 
                               tgt_dir='/home/siri/backup_on8june/Documents/Projects/Interspeech_2019/Data/baseline_40_samples/output_full')

datass = DataLoader(eng_data, batch_size=4, shuffle=True)

#print("eng data len", [ p for p in eng_data[0]])
for slt, bdl in datass:

#    print("i is", i)
    #slt = data
    print("slt is", numpy.shape(slt), "bdl is", numpy.shape(bdl))

#for i in range(len(eng_data)):
#    print("i is", i)
#    sample = eng_data[i]

 #   print(i, "slt is", sample['source_slt'].shape,"bdl is", sample['target_bdl'].shape)


