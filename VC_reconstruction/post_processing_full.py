import math 
import os


source_folder='/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Project_TAGHIM/siri_Tacotron_may2019/scripts/SGCM/VAE/VC_reconstruction/predicted/'
new_folder = '/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Project_TAGHIM/siri_Tacotron_may2019/scripts/SGCM/VAE/VC_reconstruction/postprocessed_mgc/'
os.chdir(source_folder)
files = sorted(os.listdir('.'))
for file in files:
# if file.startswith('arctic'):
  print("processing....", file)
  src = []
  f_src = open('/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1/feats/mgc/'+file.split('.')[0] + '.mgc_ascii')
  for line in f_src:
   line = line.split('\n')[0]
   src.append(line)

  f_array =[]
  f0 = open('/home3/srallaba/projects/siri_expts/Tacotron/expts_20may2019/Project_TAGHIM/siri_Tacotron_may2019/scripts/SGCM/VAE/VC_reconstruction/valid_f0/' + file.split('.')[0] + '.f0_ascii')
  for line in f0:
   line = line.split('\n')[0]
   f_array.append(line)


  p_array = []
  pred = open(file)
  for line in pred:
    line = line.split('\n')[0]
    p_array.append(line)


  g = open(new_folder + file, 'w')

  for i in range(0,len(f_array)):
#    if (f_array[i]) == -1e+10:
   if not int(float(f_array[i]))> 0:
#      print("s",f_array[i])
      g.write(str(src[i]) + '\n')
   else:
      g.write(str(p_array[i])+'\n')

  g.close()
