#!/bin/bash
folder1="/home/siri/Documents/Projects/Project_Lombard_effect/parallel_VC_data/test/"
folder2="/home/siri/Documents/Projects/Project_Lombard_effect/Data/Test/rss_arctic/feats"
folder3="/home/siri/Documents/Projects/Project_Lombard_effect/parallel_VC_data/VC_reconstruction/f0"
#mkdir $new_folder
cd $folder1
for file in *;
do
          echo $file
          fbname=$(basename "$file" .mgc_ascii)          
          cp -r  $folder2/$fbname.f0_ascii $folder3/$fbname.f0_ascii
 
done

