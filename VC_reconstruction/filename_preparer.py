#!/usr/bin/python

import os, sys
import numpy as np

# Locations
data_dir = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Project_TAGHIM/siri_Tacotron_may2019/scripts/SGCM/VAE/VC_reconstruction/valid_f0/'

g = open('files','w')

files = sorted(os.listdir(data_dir))

for file in files:
   
    fname = file.split('.')[0]
    print("writing", fname)
    g.write(str(fname)+'\n')
g.close()





