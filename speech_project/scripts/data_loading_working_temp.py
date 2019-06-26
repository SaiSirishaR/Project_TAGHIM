from torch.utils.data import Dataset, DataLoader
import os 
import numpy

class arctic_database(Dataset):

   def __init__(self, src_dir, tgt_dir):
    
    print("i got src dir as", src_dir)
    self.src_dir = src_dir
    self.tgt_dir = tgt_dir
    self.files = sorted(os.listdir(self.src_dir)) 
 
   def __len__(self):

    return len(self.files)

   def __getitem__(self, idx): 

    src_files = (sorted(os.listdir(self.src_dir))[idx])
    tgt_files = (sorted(os.listdir(self.tgt_dir))[idx])

    src_filename = os.path.join(self.src_dir,src_files)   
    print("source filename", src_filename)
    tgt_filename = os.path.join(self.tgt_dir,tgt_files)
    
    src_info = numpy.loadtxt(src_filename)
    tgt_info = numpy.loadtxt(tgt_filename)
    return src_info[idx], tgt_info[idx]



eng_data = arctic_database(src_dir='/home3/srallaba/projects/siri_expts/vocal_loudness_expts/Data/stage_1_feats/input_full', 
                               tgt_dir='/home3/srallaba/projects/siri_expts/vocal_loudness_expts/Data/stage_1_feats/output_full')

datass = DataLoader(eng_data, batch_size=4, shuffle=True)

for slt, bdl in datass:

    print("slt is", numpy.shape(slt.data), "bdl is", numpy.shape(bdl.data))
