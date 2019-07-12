import math
import numpy as np
import os


folder ='/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Project_TAGHIM/siri_Tacotron_may2019/scripts/SGCM/VAE/VC_reconstruction/src_f0/'
source_log_f0 = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Project_TAGHIM/siri_Tacotron_may2019/scripts/SGCM/VAE/VC_reconstruction/log_src_f0/'
os.chdir(folder)
files = sorted(os.listdir('.'))
for file in files:

 if file.endswith('.f0_ascii'):
  g = open(source_log_f0 + file,'w')
  f = open(file)

  for line in f:
   line = line.split('\n')[0]
   if int(float(line)) > 0:
#    print(np.log(int(float(line))))
    g.write(str(np.log(int(float(line))))+'\n')
   else:
    g.write(str(line)+'\n')
  g.close()
