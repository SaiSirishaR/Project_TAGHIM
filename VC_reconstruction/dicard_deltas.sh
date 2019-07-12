#!/bin/bash
folder="/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/reconstruction/predicted/"
#new_folder="/home/siri/Documents/Projects/NUS_projects/TTS-Male-BC2010text/wav_modified/"

#mkdir $new_folder
cd $folder
for file in *;
do
          echo $file
          fbname=$(basename "$file" _conv.mgc_ascii)
          cut -d ' ' -f 1-60 $file > $fbname.mgc_ascii
done

